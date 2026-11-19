"""Check contract versions against the merge target."""

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

REGISTRY = "src/sequifier/contract_versions.py"


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


def validate_versions(current, base):
    for name, value in current.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        previous = base.get(name)
        if (
            isinstance(previous, int)
            and not isinstance(previous, bool)
            and value < previous
        ):
            raise ValueError(f"{name} cannot decrease from {previous} to {value}")


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
    if metadata.get("tensor_payload_kind") != "tabular":
        errors.append(f"{metadata_path}: tensor_payload_kind must be tabular")
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
    validate_versions(current, base)
    validate_boundaries(root, current)


if __name__ == "__main__":
    main()
