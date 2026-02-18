from pathlib import Path
from urllib.parse import quote

BASE = "https://raw.githubusercontent.com/feraco/motiondata/main"
ROOT = Path(__file__).resolve().parent


def to_raw_url(path: Path) -> str:
    rel = path.as_posix().lstrip("./")
    return f"{BASE}/{quote(rel, safe='/')}"


def main() -> None:
    dance_files = sorted((ROOT / "dance_and_performance_movements").glob("*.csv"))
    walk_files = sorted(
        p
        for p in ROOT.rglob("*.csv")
        if "walk" in p.stem.lower() and ".git" not in p.parts
    )

    (ROOT / "dance_category_links.txt").write_text(
        "\n".join(to_raw_url(p) for p in dance_files) + ("\n" if dance_files else ""),
        encoding="utf-8",
    )

    (ROOT / "walk_dataset_links.txt").write_text(
        "\n".join(to_raw_url(p) for p in walk_files) + ("\n" if walk_files else ""),
        encoding="utf-8",
    )

    combined_lines = ["## DANCE"]
    combined_lines.extend(to_raw_url(p) for p in dance_files)
    combined_lines.append("")
    combined_lines.append("## WALK")
    combined_lines.extend(to_raw_url(p) for p in walk_files)
    (ROOT / "dance_and_walk_links.txt").write_text(
        "\n".join(combined_lines) + "\n", encoding="utf-8"
    )

    print(f"dance_count={len(dance_files)}")
    print(f"walk_count={len(walk_files)}")


if __name__ == "__main__":
    main()
