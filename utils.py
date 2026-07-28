import os

def read_market_from_file(path: str) -> list[str]:
    if not os.path.exists(path):
        return "File does not exist!"
    
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        
    return lines

def log_to_file(market: str, snippets: list):
    with open("raw_searches.txt", "a", encoding="utf-8") as f:
        f.write(f"=== {market.upper()} ===\n")
        f.write(snippets)
        f.write("\n========================\n\n")