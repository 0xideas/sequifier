"""Check global contract allocations against the target commit of a merge."""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

REGISTRY = "src/sequifier/contract_versions.py"
NON_CONTRACTS = {"HISTORICAL_MIN", "HISTORICAL_MAX", "ONNX_OPSET", "RETIRED"}

# Put explicit old-name to new-name mappings here when preserving a contract ID.
RENAMES: dict[str, str] = {}


def registry(source):
    definition = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "ContractVersions"
    )
    values = {}
    for node in definition.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            raise ValueError("ContractVersions requires plain class assignments")
        name = node.targets[0].id
        if name in values:
            raise ValueError(f"Duplicate ContractVersions attribute {name}")
        values[name] = ast.literal_eval(node.value)
    return values


def validate_allocations(current, base, renames=RENAMES):
    if current.get("HISTORICAL_MIN") != 0 or current.get("HISTORICAL_MAX") != 99:
        raise ValueError("The historical contract reservation must remain 0–99")
    active = {
        name: value for name, value in current.items() if name not in NON_CONTRACTS
    }
    retired = current["RETIRED"]
    if not isinstance(retired, tuple) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in retired
    ):
        raise ValueError("RETIRED must be a tuple of integer IDs")
    if not active or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= current["HISTORICAL_MAX"]
        for value in (*active.values(), *retired)
    ):
        raise ValueError(
            "Active and retired IDs must be integers above the historical range"
        )
    if len(set(active.values())) != len(active):
        raise ValueError("Duplicate active contract IDs")
    if len(set(retired)) != len(retired) or set(retired) & set(active.values()):
        raise ValueError("Duplicate or reused retired contract IDs")

    old_active = {
        name: value for name, value in base.items() if name not in NON_CONTRACTS
    }
    old_retired = set(base.get("RETIRED", ()))
    if not old_retired <= set(retired):
        raise ValueError("Retirement records may only be added")
    high_watermark = max(current["HISTORICAL_MAX"], *old_active.values(), *old_retired)
    for value in set(retired) - old_retired - set(old_active.values()):
        if value <= high_watermark:
            raise ValueError(
                f"Retired ID {value} was never allocated on the target branch"
            )
    preserved: set[str] = set()
    for name, old_id in old_active.items():
        replacement = renames.get(name)
        if active.get(name) == old_id:
            preserved.add(name)
        elif (
            replacement is not None
            and name not in active
            and replacement not in old_active
            and active.get(replacement) == old_id
        ):
            preserved.add(replacement)
        elif old_id not in retired:
            raise ValueError(f"{name} must retire ID {old_id}")
    for name, value in active.items():
        if name not in preserved and value <= high_watermark:
            raise ValueError(
                f"{name} allocates ID {value} at or below target high-water mark {high_watermark}"
            )


VERSION_FIELDS = {
    "version",
    "format_version",
    "schema_version",
    "manifest_version",
    "stored_window_layout_version",
    "tensor_payload_version",
    "protocol_version",
    "initialization_policy_version",
    "initialization_seed_derivation_version",
    "opset_version",
}


POSITIONAL_VERSION_FIELDS = {
    "StoredWindowLayout": (2, "version"),
    "ModelArtifact": (0, "format_version"),
}


def assignment_name(target):
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
        return target.slice.value
    return None


def has_integer_literal(value):
    if isinstance(value, ast.IfExp):
        return has_integer_literal(value.body) or has_integer_literal(value.orelse)
    return isinstance(value, ast.Constant) and (
        isinstance(value.value, int)
        and not isinstance(value.value, bool)
        or isinstance(value.value, str)
        and value.value.isdecimal()
    )


def hardcoded_versions(tree):
    """Find literal versions at Python serialization and declaration sites."""
    for node in ast.walk(tree):
        assignments = ()
        if isinstance(node, ast.Dict):
            assignments = (
                (getattr(key, "value", None), value)
                for key, value in zip(node.keys, node.values)
            )
        elif isinstance(node, ast.Assign):
            assignments = (
                (assignment_name(target), node.value) for target in node.targets
            )
        elif isinstance(node, ast.AnnAssign):
            name = assignment_name(node.target)
            assignments = ((name, node.value),)
            if name in VERSION_FIELDS and isinstance(node.value, ast.Call):
                assignments += tuple(
                    (name, argument.value)
                    for argument in node.value.keywords
                    if argument.arg == "default"
                )
            if (
                isinstance(name, str)
                and name in VERSION_FIELDS
                and isinstance(node.annotation, ast.Subscript)
                and isinstance(node.annotation.value, ast.Name)
                and node.annotation.value.id == "Literal"
            ):
                annotation = node.annotation.slice
                candidates = (
                    annotation.elts
                    if isinstance(annotation, ast.Tuple)
                    else [annotation]
                )
                assignments += tuple((name, candidate) for candidate in candidates)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in POSITIONAL_VERSION_FIELDS
        ):
            index, name = POSITIONAL_VERSION_FIELDS[node.func.id]
            if index < len(node.args):
                assignments = ((name, node.args[index]),)
        elif isinstance(node, ast.keyword):
            assignments = ((node.arg, node.value),)
        for name, value in assignments:
            if (
                isinstance(node, (ast.stmt, ast.expr, ast.keyword))
                and (
                    name in VERSION_FIELDS
                    or isinstance(name, str)
                    and name.endswith("_VERSION")
                )
                and has_integer_literal(value)
            ):
                yield node.lineno
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.search(r"^sequifier[.-][\w.-]+[.-]v[0-9]+(?:$|:)", node.value)
        ):
            yield node.lineno
        if isinstance(node, ast.FunctionDef) and node.name == "protocol_version":
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Return)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, int)
                    and not isinstance(statement.value.value, bool)
                ):
                    yield statement.lineno


def validate_boundaries(root, values):
    errors = []
    for directory in ("src/sequifier", "documentation/example_inputs", "tools"):
        for path in (root / directory).rglob("*.py"):
            if path.relative_to(root).as_posix() in {
                REGISTRY,
                "tools/check_contract_versions.py",
            }:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            errors.extend(
                f"{path.relative_to(root)}:{line}: reference ContractVersions"
                for line in hardcoded_versions(tree)
            )

    identities = {
        "sequifier.categorical-control": values["CATEGORICAL_CONTROL"],
        "sequifier.toy-control": values["TOY_CONTROL"],
    }
    for directory in ("src/sequifier/templates", "documentation/example_inputs"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix not in (".yaml", ".yml", ".json"):
                continue
            content = path.read_text()
            for prefix, version in identities.items():
                expected = f"{prefix}.v{version}"
                for match in re.finditer(rf"{re.escape(prefix)}\.v[\w.-]+", content):
                    if match[0] != expected:
                        errors.append(
                            f"{path.relative_to(root)}: found {match[0]}, expected {expected}"
                        )
    metadata_path = (
        "documentation/example_inputs/configs/ddconfigs/preprocessing_input.json"
    )
    metadata = json.loads((root / metadata_path).read_text())
    for field, name in (
        ("tensor_payload_version", "DATASET_TENSOR_METADATA"),
        ("stored_window_layout_version", "STORED_WINDOW_LAYOUT"),
    ):
        actual = metadata.get(field)
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual != values[name]
        ):
            errors.append(
                f"{metadata_path}: {field} must match ContractVersions.{name}"
            )
    if errors:
        raise ValueError("\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Target commit included in HEAD")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.base, "HEAD"], cwd=root, check=True
    )
    base_paths = subprocess.check_output(
        ["git", "ls-tree", "--name-only", args.base, REGISTRY], text=True, cwd=root
    )
    base = (
        registry(
            subprocess.check_output(
                ["git", "show", f"{args.base}:{REGISTRY}"], text=True, cwd=root
            )
        )
        if base_paths
        else {}
    )
    current = registry((root / REGISTRY).read_text())
    validate_allocations(current, base)
    validate_boundaries(root, current)


if __name__ == "__main__":
    main()
