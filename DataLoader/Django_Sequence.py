from __future__ import annotations
import os
import django
from typing import Any
from types import SimpleNamespace
from django.db.models import QuerySet
from abc import ABC, abstractmethod
from django.db import close_old_connections
from .SequenceBase import SequenceBase, T_Data


_DJANGO_SETTINGS_DEFAULT = "annotationserver.core.settings.development"

def ensure_django():
    # Override the settings module via MACVO_DJANGO_SETTINGS or DJANGO_SETTINGS_MODULE env vars.
    # On non-EIVA machines, set MACVO_DJANGO_SETTINGS to the appropriate Django settings module.
    if not os.environ.get("DJANGO_SETTINGS_MODULE"):
        os.environ["DJANGO_SETTINGS_MODULE"] = os.environ.get("MACVO_DJANGO_SETTINGS", _DJANGO_SETTINGS_DEFAULT)
        django.setup()
    close_old_connections()


class DjangoORMSequence(SequenceBase[T_Data], ABC):
    def __init__(self, config: dict | Any):
        ensure_django()
        self.cfg = self.config_dict2ns(config)

        # Build a base queryset. Do Not materialize rows yet
        base_qs = self.build_queryset(self.cfg)

        # Freeze ordering & IDs up-front so __len__/indexing is O(1) and stable.
        # Avoids OFFSET/LIMIT per item.
        ordering = self.default_ordering()
        if ordering:
            # self._pkl_list: list[int] = list(
            #     base_qs.order_by(self.default_ordering).values_list("pk", flat=True)
            # )
            base_qs = base_qs.order_by(*(
                ordering if isinstance(ordering, (list, tuple)) else [ordering]
                ),
                "pk"
            )

        # snapshot PKs so indexing is O(1) and stable
        self._pk_list = list(base_qs.values_list("pk", flat=True))
        lenpk = len(self._pk_list)
        super().__init__(lenpk)
        # build a fetch-optimized queryset AFTER init
        self._fetch_qs = self.optimize_fetch(self.build_queryset(self.cfg))

        # optional caches
        self.preload_side_data(self.cfg)

    @abstractmethod
    def build_queryset(self, config: dict | Any) -> QuerySet:
        ...

    @abstractmethod
    def record_to_frame(self, row: Any, *, local_index: int, original_index: int) -> T_Data:
        ...

    def default_ordering(self) -> str:
        """
        Default ordering for the queryset.
        Override this method to specify how the records should be ordered.
        """
        return "pk"

    def optimize_fetch(self, queryset: QuerySet) -> QuerySet:
        """
        Override this method to optimize the queryset.
        For example, you can use `select_related` or `prefetch_related`
            to reduce database hits.
        """
        return queryset

    def preload_side_data(self, cfg: SimpleNamespace) -> None:
        pass  # override to cache things like classifications, sequence cameras, etc.

    # ---------- core retrieval ----------
    def __getitem__(self, local_index):
        orig_index = self.get_index(local_index)
        pk = self._pk_list[orig_index]
        row = self._fetch_qs.get(pk=pk)
        return self.record_to_frame(row, local_index=local_index, original_index=orig_index)

    # ---------- worker friendliness ----------
    def __getstate__(self):
        # QuerySets & DB connections aren’t picklable; rebuild them in workers.
        state = self.__dict__.copy()
        state.pop("_fetch_qs", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        ensure_django()
        # Rebuild fetch queryset from the same config & model
        base_qs = self.build_queryset(self.cfg)
        self._fetch_qs = self.optimize_fetch(base_qs)
