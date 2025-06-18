import click

@click.group()
def cli():
    pass

@cli.command()
def example_function():
    """Does something"""
    # function()


if __name__ == '__main__':
    # Run with `python3 -m src.main`, see options with `python3 -m src.main --help`
    cli()
