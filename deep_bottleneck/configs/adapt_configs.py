from pathlib import Path
import json
import shutil


def main():
    new_base_dir = Path("cohort_17_paper/")
    shutil.rmtree(new_base_dir)
    for p in Path("cohort_16_paper/").glob("**/*.json"):
        print(p)
        d = read_json(p)
        d = change(d)
        new_path = new_base_dir.joinpath(*p.parts[1:])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(new_path, d)

        
def read_json(p):
    return json.loads(p.read_text())
        
    

def write_json(p, d):
    return p.write_text(json.dumps(d, indent=4))    


def change(d):
    d["epochs"] = 10_000
    d["n_runs"] = 10
    return j

if __name__ == "__main__":
    main()