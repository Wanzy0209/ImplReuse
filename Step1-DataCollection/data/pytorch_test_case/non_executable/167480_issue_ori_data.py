_ @sumantro93 your example highlights the point I was trying to make: it's not framework's responsibility to sanitize the inputs.

For example, In the sample that you've shared, one is allowed to compile and run any untrusted code, which is a huge security issue on its own, so even if this issue is fixed, one is already allowed to execute arbitrary code on the host by the Flask endpoint developer.

Closing, but please do not hesitate to report it as regular issue or propose a pull request that would sanitize the inputs _