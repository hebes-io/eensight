from pathlib import Path

from kedro.framework.project import configure_project

from .cli import cli


def main():
    configure_project(Path(__file__).parent.name)
    cli()


if __name__ == "__main__":
    main()
