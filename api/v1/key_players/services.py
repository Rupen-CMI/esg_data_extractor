import os, asyncio, random, json, logging, time, re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("pipeline")
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select
from ..models import Market, Company, MarketCompanyLink
from dotenv import load_dotenv
from ddgs import DDGS
from zen_client import call_with_prompt, DEFAULT_MODEL

load_dotenv()

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "cmi_reports_202606241500.txt")
ASYNC_DB_URL = os.getenv("ASYNC_DB_URL")

async_engine = create_async_engine(
    ASYNC_DB_URL,
    pool_size=15,          # Caps connections so 10k loops don't overwhelm Postgres
    max_overflow=5,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

BATCH_SIZE = 10

def _parse_market_names(path: str) -> list[str]:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("|") or stripped.startswith("|---"):
                continue
            name = stripped.strip("|").strip()
            if name and name.lower() != "keyword":
                names.append(name)
    return names


def seed_markets(db: Session) -> dict:
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    names = _parse_market_names(DATA_FILE)
    if not names:
        return {"inserted": 0, "skipped": 0, "total_parsed": 0}

    rows = [{"name": n, "status": "pending"} for n in names]

    stmt = insert(Market).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["name"])
    result = db.execute(stmt)
    db.commit()

    inserted = result.rowcount
    skipped = len(names) - inserted

    return {
        "inserted": inserted,
        "skipped": skipped,
        "total_parsed": len(names),
    }

async def run_pipeline():
    log.info("Pipeline started.")
    batch_num = 0
    pipeline_start = time.perf_counter()
    while True:
        batch_num += 1
        log.info(f"--- Batch #{batch_num} ---")
        batch_start = time.perf_counter()
        has_more = await process_batch()
        elapsed = round(time.perf_counter() - batch_start, 1)
        log.info(f"Batch #{batch_num} finished in {elapsed}s")
        if not has_more:
            total = round(time.perf_counter() - pipeline_start, 1)
            log.info(f"All markets processed. Total time: {total}s")
            break
        rest = random.randint(5, 10)
        log.info(f"Resting {rest}s before next batch...")
        await asyncio.sleep(rest)


async def process_batch():
    try:
        async with AsyncSessionLocal() as session:
            query = (
                select(Market)
                .where(Market.status == "pending")
                .limit(BATCH_SIZE)
            )
            result = await session.execute(query)
            markets = result.scalars().all()

            if not markets:
                log.info("No pending markets left.")
                return False

            log.info(f"Claimed {len(markets)} markets: {[m.name for m in markets]}")
            for market in markets:
                market.status = "processing"
            await session.commit()

            tasks = [fetch_and_extract(market.name, i) for i, market in enumerate(markets)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            done, failed = 0, 0
            for market, companies in zip(markets, results):
                if isinstance(companies, Exception) or not companies:
                    market.status = "failed"
                    err = companies if isinstance(companies, Exception) else "no data"
                    log.warning(f"Failed [{market.name}]: {err}")
                    failed += 1
                else:
                    company_list = companies["companies"]
                    if not company_list:
                        market.status = "no_results"
                        log.warning(f"[{market.name}] LLM returned empty company list")
                    else:
                        await save_companies_to_db(session, market, company_list)
                        market.status = "done"
                        log.info(f"Saved [{market.name}]: {len(company_list)} companies")
                    done += 1

            await session.commit()
            log.info(f"Batch complete — done: {done}, failed: {failed}")
            return True
    except Exception as e:
        log.error(f"Batch crashed: {e}")
        return False 


async def fetch_and_extract(market_name: str, index: int) -> dict | None:
    await asyncio.sleep(index * 2)
    
    log.info(f"[{market_name}] Starting DDGS search...")
    t0 = time.perf_counter()
    context = await fetch_players_ddgs(market_name)
    
    if not context or not context.strip():
        log.warning(f"[{market_name}] DDGS returned empty result")
        return None
    
    log.info(f"[{market_name}] DDGS done in {round(time.perf_counter() - t0, 1)}s — sending to LLM...")
    t1 = time.perf_counter()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: extract_key_players(market_name, context))
    log.info(f"[{market_name}] LLM done in {round(time.perf_counter() - t1, 1)}s")
    
    return result

async def fetch_by_market(market_name: str) -> dict:
    """
    Fetch key players for a single market by name, upsert them into the DB,
    and link them to the market. Creates the market row if it doesn't exist.
    """
    async with AsyncSessionLocal() as session:
        # Get or create market
        result = await session.execute(select(Market).where(Market.name == market_name))
        market = result.scalar_one_or_none()

        if not market:
            market = Market(name=market_name, status="processing")
            session.add(market)
            await session.flush()
        else:
            market.status = "processing"

        await session.commit()
        await session.refresh(market)

    # Fetch + extract outside the session to avoid holding a connection during slow IO
    extracted = await fetch_and_extract(market_name, index=0)

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Market).where(Market.id == market.id))
        market = result.scalar_one()

        if not extracted or not extracted.get("companies"):
            market.status = "no_results"
            await session.commit()
            return {"status": "fail", "error": f"No companies found for '{market_name}'"}

        company_list = extracted["companies"]
        await save_companies_to_db(session, market, company_list)
        market.status = "done"
        await session.commit()

    log.info(f"[fetch_by_market] '{market_name}' → {len(company_list)} companies saved")
    return {
        "status": "success",
        "market": market_name,
        "companies_found": len(company_list),
        "companies": company_list,
    }


async def fetch_players_ddgs(market: str):
    delay = random.uniform(3, 6)
    log.debug(f"[{market}] Pre-search delay {round(delay, 1)}s")
    await asyncio.sleep(delay)

    search_query = f"{market}'s Top 10 key player companies list"
    snippets = []

    try:
        loop = asyncio.get_event_loop()
        with DDGS() as ddgs:
            results = await loop.run_in_executor(
                None, lambda: list(ddgs.text(search_query, max_results=3))
            )
            for r in results:
                snippets.append(r.get("body") or "")
        log.debug(f"[{market}] Got {len(snippets)} snippets")
    except Exception as e:
        log.warning(f"[{market}] DDGS failed: {e}")

    delay = random.uniform(3, 6)
    log.debug(f"[{market}] Post-search delay {round(delay, 1)}s")
    await asyncio.sleep(delay)

    return "\n".join(snippets)


def extract_key_players(market: str, context: str) -> list | None:
    prompt = f"""You are a precise data extraction engine.
        From the search results below, extract the top key player companies for this market.

        Return ONLY a valid JSON array, no prose, no markdown:
        [
        {{"name": "Company Name", "country": "Country of headquarters", "confidence": "high | medium | low"}}
        ]

        Rules:
        - confidence "high" = explicitly named as key player
        - confidence "medium" = mentioned as a company in this market
        - confidence "low" = inferred or unclear
        - Do NOT invent companies not present in the text
        - country = headquarters country. If not mentioned in the text, use your own knowledge to fill it in. Never leave country empty or null.
        - Max 10 companies

        Market: {market}
        Search results:
        \"\"\"{context}\"\"\"
    """

    result = call_with_prompt(prompt, model=DEFAULT_MODEL, max_tokens=4000)
    if not result["ok"]:
        log.error(f"[{market}] LLM call failed: {result['error']}")
        return None

    raw = result["raw"].strip()

    # Strip markdown code fences if present
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    # Try direct parse first
    try:
        companies = json.loads(raw)
        if isinstance(companies, list):
            return {"market": market, "companies": companies}
    except Exception:
        pass

    # Fallback: extract JSON array from within reasoning text
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        try:
            companies = json.loads(match.group(0))
            if isinstance(companies, list):
                return {"market": market, "companies": companies}
        except Exception:
            pass

    log.error(f"[{market}] JSON parse failed. Raw response:\n{raw[:500]}")
    return None


async def save_companies_to_db(session: AsyncSession, market: Market, companies: list):
    names = [c["name"] for c in companies if c.get("name")]
    if not names:
        return

    # Upsert companies — skip duplicates by name
    stmt = insert(Company).values([
        {"name": c["name"], "country": c.get("country")}
        for c in companies if c.get("name")
    ])
    stmt = stmt.on_conflict_do_nothing(index_elements=["name"])
    await session.execute(stmt)

    # Fetch IDs for all company names (includes pre-existing ones)
    result = await session.execute(select(Company).where(Company.name.in_(names)))
    fetched = result.scalars().all()

    # Upsert market-company links
    if fetched:
        link_stmt = insert(MarketCompanyLink).values([
            {"market_id": market.id, "company_id": company.id}
            for company in fetched
        ])
        link_stmt = link_stmt.on_conflict_do_nothing()
        await session.execute(link_stmt)