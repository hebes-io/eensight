# -*- coding: utf-8 -*-

"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal

from eensight.config.config import OmegaConfigLoader


class ProjectHooks:
    @hook_impl
    def register_config_loader(
        self,
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> OmegaConfigLoader:
        return OmegaConfigLoader(
            conf_paths,
            globals_pattern=["globals*", "globals*/**", "**/globals*"],
            merge_keys=["rebind_names", "sources"],
        )

    @hook_impl
    def register_catalog(
        self,
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )


class DataCatalogHooks:
    pass
