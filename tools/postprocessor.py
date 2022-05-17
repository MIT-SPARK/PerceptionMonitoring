import click
from pathlib import Path
from enum import Enum
from tools.fault_identification import postprocess as fault_identification

class OpResult:
    class Type(Enum):
        SUCCESS = 0
        FAILURE = 1
        SKIP = 2

    def __init__(self, type: Type, message: str = None):
        self.type = type
        self.message = message


@click.group()
def cli():
    pass

@cli.command()
# fmt: off
@click.option("--runid", type=str, required=True, help="Run id")
@click.option("--dir", type=Path, required=True, help="Config file")
@click.option("--config", type=Path, required=True, help="Config file")
@click.option("--input", type=str, required=True, help="Log filename")
@click.option("--output", type=str, required=True, help="Output folder")
@click.option("--force", is_flag=True, default=False, help="Force reprocess")
# fmt: on
def postprocess(runid, dir, config, input, output, force):
    if not dir.is_dir():
        return OpResult(OpResult.Type.FAILURE, "Dir path is not a directory.")
    for d in dir.iterdir():
        if d.is_dir():
            print(f"Preprocessing {d}")
            r = fault_identification(
                run_id=runid, 
                config=config, 
                dataset_path=d / input, 
                outfile=d / output, 
                force=force, 
                verbose=False
            )
            if r:
                print("✅")
            else:
                print("💥")
        print("")

if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
    # print("Waiting for debugger attach")
    # ptvsd.wait_for_attach()

    cli()
