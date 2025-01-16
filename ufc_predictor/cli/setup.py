import click
from pathlib import Path
from ufc_predictor.config import PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR

@click.command()
@click.option('--force/--no-force', default=False, help='Force recreation of directories')
def setup_project(force):
    """Set up the initial project structure."""
    dirs = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]
    
    for dir_path in dirs:
        if force and dir_path.exists():
            click.echo(f"Removing existing directory: {dir_path}")
            for file in dir_path.glob('*'):
                file.unlink()
            dir_path.rmdir()
        
        if not dir_path.exists():
            click.echo(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True)
        else:
            click.echo(f"Directory already exists: {dir_path}")

if __name__ == "__main__":
    setup_project() 