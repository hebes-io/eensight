from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.startup import (
    ProjectMetadata,
    _add_src_to_path,
    _get_project_metadata,
)


def bootstrap_project(project_path: Path) -> ProjectMetadata:
    """Run setup required at the beginning of the workflow
    when running in project mode, and return project metadata.
    """
    metadata = _get_project_metadata(project_path)
    metadata = ProjectMetadata(
        config_file=metadata.config_file,
        package_name=metadata.package_name,
        project_name=metadata.project_name,
        project_path=metadata.project_path,
        project_version=metadata.project_version,
        source_dir=Path(metadata.project_path).expanduser(),
    )

    _add_src_to_path(metadata.source_dir, project_path)
    configure_project(metadata.package_name)
    return metadata
