# Contract versions

Identify each contract independently of its version number. When a reader handles multiple contracts, select the contract using an explicit discriminator such as a format tag, mode, payload kind, or protocol name. A dedicated reader or field can establish the contract through its schema context.

Validate the version only after identifying the contract. Different contracts may share the same version number. Version changes and CI checks must be scoped to each contract.
