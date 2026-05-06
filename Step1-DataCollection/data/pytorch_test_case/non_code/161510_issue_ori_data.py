++ grep -v '?? third_party'
+ git_status=' M third_party/kineto'
Build left local git repository checkout dirty
git status --porcelain:
 M third_party/kineto
+ [[ -n  M third_party/kineto ]]
+ echo 'Build left local git repository checkout dirty'
+ echo 'git status --porcelain:'
+ echo ' M third_party/kineto'