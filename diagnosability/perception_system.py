from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Iterable, Optional
import re
from enum import Enum
import yaml
from abc import ABC, abstractmethod


class ModuleStateOracle(ABC):
    @abstractmethod
    def __call__(self, evidence):
        """Given the output states, compute the module state."""
        pass


class DefaultDatatype(Enum):
    NUMERIC = 0
    REL_POSE = 1
    POSE = 2
    FEATURES_2D = 3
    FEATURES_3D = 4
    POINT_CLOUD = 5
    TRAFFIC_LIGHTS = 6
    LANES = 7
    OBSTACLES = 8
    UNKNOWN = 9


@dataclass(eq=True)
class NamedObject:
    name: str
    uuid: Optional[str] = None

    @property
    def varname(self):
        return f"{self.uuid}_{self.name}"

    def __post_init__(self):
        not_allowed = set("-,.<>")
        assert not any(
            (c in not_allowed) for c in self.name
        ), f"Name cannot contain `{not_allowed}`"
        if self.uuid is None:
            self.update_uuid()

    def update_uuid(self):
        self.uuid = str(id(self))


class Collection(dict, Iterable[NamedObject]):
    def __init__(self, elements: List[NamedObject] = []):
        super().__init__({u.name: u for u in elements})
        assert len(elements) == len(self), "Duplicate keys."

    def __iter__(self):
        return iter(self.values())

    def __repr__(self):
        return str(list(self.values()))

    def add(self, element: NamedObject):
        assert element.name not in self, "Duplicate keys."
        self[element.name] = element

    def copy(self):
        return deepcopy(self)


@dataclass(eq=True)
class FailureMode(NamedObject):
    severity: int = 0


@dataclass(eq=True)
class Unit(NamedObject):
    """Base class for a unreliable system unit (e.g. a module, output, sensor etc)"""
    failure_modes: List = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.failure_modes = Collection(self.failure_modes)

    @property
    def cardinality(self):
        return len(self.failure_modes)


@dataclass(eq=True)
class Test(NamedObject):
    scope: List[FailureMode] = field(default_factory=list)
    timestep: Optional[int] = None


@dataclass(eq=True)
class Output(Unit):
    datatype: Enum = DefaultDatatype.UNKNOWN


@dataclass(eq=True)
class Module(Unit):
    inputs: List[Output] = field(default_factory=list)
    outputs: List[Output] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        # self.inputs = Collection(self.inputs)
        self.outputs = Collection(self.outputs)


class FunctionalModuleState(ModuleStateOracle):
    def __init__(self, relations):
        self.relations = relations

    def __call__(self, evidence):
        return {fm: int(fcn(evidence)) for fm, fcn in self.relations.items()}


class System(Collection):
    class Filter:
        MODULE_ONLY = 0
        OUTPUT_ONLY = 1
        ALL = 2

    def __init__(self, modules: List[Module]):
        super().__init__(modules)
        self.relations = None

    @classmethod
    def from_yaml(cls, yaml_file):
        fmode_rex = re.compile(r"([\w\s*]+)+\((\d+)\)")

        def parse_fmode(string):
            result = fmode_rex.search(string)
            if result is not None:
                return FailureMode(
                    name=result.group(1).strip(), severity=int(result.group(2))
                )
            else:
                return FailureMode(name=string, severity=0)

        with open(yaml_file, "r") as stream:
            cfg = yaml.safe_load(stream)
        data = dict()
        modules = []
        for o in cfg["Data"]:
            out = cfg["Data"][o]
            fmodes = [parse_fmode(f) for f in out["FailureModes"]]
            type = DefaultDatatype[out["Type"].upper()]
            data[o] = Output(name=out["Name"], datatype=type, failure_modes=fmodes)
        for m in cfg["Modules"]:
            mod = cfg["Modules"][m]
            fmodes = [parse_fmode(f) for f in mod["FailureModes"]]
            if "Inputs" in mod:
                module_inputs = [data[o] for o in mod["Inputs"]]
            else:
                module_inputs = []
            if "Outputs" in mod:
                module_outputs = [data[o] for o in mod["Outputs"]]
            else:
                module_outputs = []
            modules.append(
                Module(
                    name=mod["Name"],
                    inputs=module_inputs,
                    outputs=module_outputs,
                    failure_modes=fmodes,
                )
            )
        sys = cls(modules)
        if "Relations" in cfg:
            # gt_relations = dict()
            # for rel in cfg["GTRelations"]:
            #     varname = sys.query(rel["FailureMode"]).varname
            #     scope = [sys.query(f).varname for f in rel["If"]]
            #     if rel["Type"] == "AtLeastOne":
            #         f = lambda e: any(e[f.varname] for f in scope)
            #     elif rel["Type"] == "Majority":
            #         f = lambda e: sum(e[f.varname] for f in scope) >= len(scope) // 2
            #     else:
            #         raise ValueError(f"Unknown relation type: {rel['type']}")
            #     gt_relations[varname] = f
            # sys.oracle = FunctionalModuleState(gt_relations)
            sys.relations = cfg["Relations"]
        return sys

    def has_oracle(self):
        return self.oracle is not None

    def oracle(self, evidence):
        if self.oracle is None:
            raise ValueError("Oracle not defined.")
        states = dict()
        for rel in self.relations:
            varname = self.query(rel["FailureMode"]).varname
            scope = [self.query(f).varname for f in rel["If"]]
            if rel["Type"] == "AtLeastOne":
                states[varname] = any(evidence[f] for f in scope)
            elif rel["Type"] == "Majority":
                states[varname] = sum(evidence[f] for f in scope) >= len(scope) // 2
            else:
                raise ValueError(f"Unknown relation type: {rel['type']}")
        return states

    def query(self, query):
        """Query the model

        Args:
            query (str): Valid queries
                - "module A": return module A
                - "module A -> *": return all outputs of `module A`
                - "module A . *": return all failure modes of `module A`
                - "module.failure x":  return `failure x` of `module A`
                - "module A -> output 1": return `output 1` of `module A`
                - "module A -> output 1 . failure mode x": return `failure mode x` of `output 1` (of `module A`)
                - "module A -> output 1 . *": return all failure modes of `output 1` (of `model A`)
        """

        class Step:
            MODULE = 0
            OUTPUT = 1
            FAILURE_MODE = 2
            PARSE = 3
            GET = 4

        if query == "*":
            return self
        pattern = "\s*(->)\s*|\s*(\.)\s*"
        expr = filter(None, re.split(pattern, query))
        step = Step.MODULE
        curr = self
        for x in expr:
            if step == Step.PARSE:
                if x == "->":
                    step = Step.OUTPUT
                elif x == ".":
                    step = Step.FAILURE_MODE
            elif step == Step.MODULE:
                curr = curr[x]
                step = Step.PARSE
            elif step == Step.OUTPUT:
                curr = curr.outputs
                step = Step.PARSE
                if x == "*":
                    curr = list(curr)
                else:
                    curr = curr[x]
            elif step == Step.FAILURE_MODE:
                curr = curr.failure_modes
                if x == "*":
                    curr = list(curr)
                else:
                    curr = curr[x]
                step = Step.PARSE
        if step != Step.PARSE:
            raise ValueError(f"Invalid query: {query}")
        if curr == self:
            curr = None
        return curr

    def rev_query(self, obj: NamedObject):
        for module in self:
            if module is obj:
                return f"{module.name}"
            for f in module.failure_modes:
                if f is obj:
                    return f"{module.name}.{f.name}"
            for o in module.outputs:
                if o is obj:
                    return f"{module.name}->{o.name}"
                for f in o.failure_modes:
                    if f is obj:
                        return f"{module.name}->{o.name}.{f.name}"
        return None

    def find_by_varname(self, varname):
        for m in self:
            if m.varname == varname:
                return m
            for f in m.failure_modes:
                if f.varname == varname:
                    return f
            for o in m.outputs:
                if o.varname == varname:
                    return o
                for f in o.failure_modes:
                    if f.varname == varname:
                        return f
        return None

    def parent(self, unit):
        for m in self:
            if m.varname == unit.varname:
                return None
            for f in m.failure_modes:
                if f.varname == unit.varname:
                    return m
            for o in m.outputs:
                if o.varname == unit.varname:
                    return m
                for f in o.failure_modes:
                    if f.varname == unit.varname:
                        return o

    def get_failure_modes(self, filter=Filter.ALL):
        fm = []
        if filter == System.Filter.MODULE_ONLY or filter == System.Filter.ALL:
            fm += [f for m in self.values() for f in m.failure_modes]
        if filter == System.Filter.OUTPUT_ONLY or filter == System.Filter.ALL:
            fm += [f for m in self.values() for o in m.outputs for f in o.failure_modes]
        return fm

    def copy(self, update_uuid=False):
        cpy = super().copy()
        if update_uuid:
            for m in cpy:
                m.update_uuid()
                for f in m.failure_modes:
                    f.update_uuid()
                for o in m.outputs:
                    o.update_uuid()
                    for f in o.failure_modes:
                        f.update_uuid()
        return cpy
