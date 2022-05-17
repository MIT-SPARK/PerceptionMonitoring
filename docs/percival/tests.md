# Weights

The class weights are used store the densities associated with factors (i.e. tests, relationships, etc.).

Suppose we have a perfect test with scope `{"f1", "f2" }`, the density table is

| t   | f1  | f2  | density |
|-:-:-|-:-:-|-:-:-|-:-:-----|
| 0   | 0   | 0   | 1.0     |
| 0   | 0   | 1   | 0.0     |
| 0   | 1   | 0   | 0.0     |
| 0   | 1   | 1   | 1.0     |
| 1   | 0   | 0   | 0.0     |
| 1   | 0   | 1   | 1.0     |
| 1   | 1   | 0   | 1.0     |
| 1   | 1   | 1   | 1.0     |

so the weights vector would be

```cpp
Weights w({1,0,0,1,0,1,1,1});
```

it is important to place the `t` as first column.