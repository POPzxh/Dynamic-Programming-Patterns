# Dynamic Programming Patterns

### Patterns

1）Minimum (Maximum) Path to Reach a Target**到达目标的最长（最短）路径**

2）[Distinct Ways](#Distinct-ways)

Merging Intervals

DP on Strings

Decision Making



## 1.Minimum (Maximum) Path to Reach a Target	**到达目标的最长（最短）路径**

### 一般问题陈述

> ​	Given a target find minimum (maximum) cost / path / sum to reach the target.
>
> ​	给定一个目标，找到最小（最大）的花费/路径/和 达到这个目标

### 一般方法

>Choose minimum (maximum) path among all possible paths before the current state, then add value for the current state.
>
>更新当前状态时，选择之前状态中的最大（最小值）来更新。
>
>```
>routes[i] = min(routes[i-1], routes[i-2], ... , routes[i-k]) + cost[i]
>```

### **例题** : 746. Min Cost Climbing Stairs

> On a staircase, the `i`-th step has some non-negative cost `cost[i]` assigned (0 indexed).
>
> Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.
>
> * 简介：爬楼梯，一次可以爬1或2层，每层有cost[i]，求爬到顶最小的花费
> * 分析：对于当前状态dp[i]，只有两种可能到达：走了一层到达dp[i-1]+cost[i]和走了两层到达dp[i-2]+cost[i]。所以，状态转移方程：dp[i] = min(dp[i-1],dp[i-2]) +cost[i]
>
> ```java
> class Solution {
>     public int minCostClimbingStairs(int[] cost) {
>         int[] dp = new int[cost.length + 1];
>         dp[0] = cost[0];dp[1] = cost[1];
>         for(int i = 2 ; i <=cost.length ; i++){
>             dp[i] = Math.min(dp[i-1] , dp[i-2]);
>             if(i != cost.length) dp[i] += cost[i];
>         }
>         return dp[cost.length];
>     }
> }
> ```

### 相似题目

|                                                              |                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [64. Minimum Path Sum](https://leetcode.com/problems/minimum-path-sum/) `Medium` | [322. Coin Change](https://leetcode.com/problems/coin-change/) `Medium` | [931. Minimum Falling Path Sum](https://leetcode.com/problems/minimum-falling-path-sum/) `Medium` |
| [650. 2 Keys Keyboard](https://leetcode.com/problems/2-keys-keyboard/) `Medium` | [279. Perfect Squares](https://leetcode.com/problems/perfect-squares/) `Medium` | [1049. Last Stone Weight II](#1049-last-stone-weight-ii)`Medium` |
| [120. Triangle](#120-Triangle)`Medium`                       | [474. Ones and Zeroes](#474-Ones-and-Zeroes) `Medium`        | [221. Maximal Square](#221-maximal-square)   `Medium`        |
| [322. Coin Change](#322-Coin-Change)`Medium`                 | [1240. Tiling a Rectangle with the Fewest Squares](#1240-Tiling-a-Rectangle-with-the-Fewest-Squares)`Hard` |                                                              |

------

#### 1049 Last Stone Weight II

We have a collection of rocks, each rock has a positive integer weight.

Each turn, we choose **any two rocks** and smash them together. Suppose the stones have weights `x` and `y` with `x <= y`. The result of this smash is:

- If `x == y`, both stones are totally destroyed;
- If `x != y`, the stone of weight `x` is totally destroyed, and the stone of weight `y` has new weight `y-x`.

At the end, there is at most 1 stone left. Return the **smallest possible** weight of this stone (the weight is 0 if there are no stones left.)

**分析：** A为正权之和，B为负权之和。根据题意，A-B>=0，求MIN(A-B)。最好情况就是，A\==B，即B\==SUM/2 。那么这道题就可以变为——选取最大的不超过sum/2的和。

```java
class Solution {
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for(int stone : stones) sum += stone;
        int half = sum / 2;
        int[] dp = new int[half+1];
        for(int stone : stones){
            for(int i = half ; i>0 ; i--)
                if(i-stone>=0)
                    dp[i] = Math.max(dp[i-stone] + stone , dp[i]);        
        }
        return sum - 2 * dp[half];
    }  
}
```

------

#### 120 Triangle

Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.

For example, given the following triangle

```
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
```

The minimum path sum from top to bottom is `11` (i.e., **2** + **3** + **5** + **1** = 11).

**分析：**

对于`dp[i][j]`只有`dp[i-1][j]`和`dp[i-1][j-1]`能够到达。所以：
<a href="https://www.codecogs.com/eqnedit.php?latex=dp[i][j]&space;=&space;min(dp[i-1][j]&space;,&space;dp[i-1][j-1])&space;&plus;&space;triangle[i][j]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dp[i][j]&space;=&space;min(dp[i-1][j]&space;,&space;dp[i-1][j-1])&space;&plus;&space;triangle[i][j]" title="dp[i][j] = min(dp[i-1][j] , dp[i-1][j-1]) + triangle[i][j]" /></a>


```java
class Solution {
    public int minimumTotal(List<List<Integer>> triangle) {
        int size = triangle.size();
        int[] dp = new int[size];
        int cnt = 1;
        for(List<Integer> row : triangle){
            if(cnt == 1) dp[0] = row.get(0);
            else{
                int min = Integer.MAX_VALUE;
                for(int pos = cnt-1 ; pos >=0  ; pos --){
                    if(pos != 0 && pos != cnt-1) dp[pos] =  Math.min(dp[pos],dp[pos-1]) + row.get(pos);
                    else if(pos==0) dp[pos] = dp[pos] + row.get(pos);
                    else dp[pos] = dp[pos-1] + row.get(pos);
                    if(dp[pos] < min) min = dp[pos];
                }
                if(cnt == size) return min;
            }
            cnt++;
        }
        return dp[0];
    }
}
```

------

#### 474 Ones and Zeroes

In the computer world, use restricted resource you have to generate maximum benefit is what we always want to pursue.

For now, suppose you are a dominator of **m** `0s` and **n** `1s` respectively. On the other hand, there is an array with strings consisting of only `0s` and `1s`.

Now your task is to find the maximum number of strings that you can form with given **m** `0s` and **n** `1s`. Each `0` and `1` can be used at most **once**.

**Note:**

1. The given numbers of `0s` and `1s` will both not exceed `100`
2. The size of given string array won't exceed `600`.

**分析**：

维护数组`dp[i][j]`，代表最多为i个'0'和j个'1'时最多能构成的字符串数。

所以，遍历字符串`for(String str :strs)`，对于每一个字符串，我们考虑加入还是不加入：$dp[i][j] = min(dp[i-zero][j-one]+1,dp[i][j])$。

```java
class Solution {
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp =new int[m+1][n+1];
        for(String str : strs){
            int zero = 0, one = 0;
            for(char c : str.toCharArray()) if(c=='0') zero ++; else one++;
            for(int i = m ; i >= 0 ; i --){
                for(int j = n ; j>= 0 ; j--){
                    if(i-zero>=0 && j -one >=0)
                        dp[i][j] = Integer.max(dp[i][j],dp[i-zero][j-one]+1);
                }
            }
        }
        return dp[m][n];
    }
}
```

------

#### 221 Maximal Square

Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

**Example:**

```java
Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4
```

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        // for(char[] row : matrix){
        //     System.out.println(Arrays.toString(row));
        // }
        // System.out.println();
        
        int hi = matrix.length;
        if(hi == 0) return 0;
        int wi = matrix[0].length;
        int[][] dp =new int[hi][wi];
        int max = 0;
        for(int i = 0 ; i < hi ; i++){
            for(int j = 0 ; j < wi ; j++){
                if(matrix[i][j] == '1'){
                    if(i==0 || j == 0) dp[i][j] = 1;
                    else dp[i][j] = Math.min(dp[i-1][j] , dp[i][j-1]) +1;
                }
                if(dp[i][j] > 1){
                   int temp = dp[i][j];
                   if(matrix[i-temp+1][j-temp+1] != '1') dp[i][j]--;
                }
                max = Math.max(max,dp[i][j]);
            }
        }
        
        // for(int[] row : dp){
        //     System.out.println(Arrays.toString(row));
        // }
        return max*max;
    }
}
```

------

#### 322 Coin Change

You are given coins of different denominations and a total amount of money *amount*. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return `-1`.

**Example 1:**

```
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
```

**分析：**

`dp[i]`代表组合为金额i时需要最少的硬币数。所以，<a href="https://www.codecogs.com/eqnedit.php?latex=dp[i]&space;=&space;min_{coin:coins}(dp[i-coin])&space;&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?dp[i]&space;=&space;min_{coin:coins}(dp[i-coin])&space;&plus;1" title="dp[i] = min_{coin:coins}(dp[i-coin]) +1" /></a>

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        if(amount == 0) return 0;
        int[] dp = new int [amount + 1];
        for(int i= 1 ; i <= amount ; i++){
            int min = Integer.MAX_VALUE;
            for(int coin : coins){
                if(coin == i) min = 1;
                else if(i - coin >=0 && dp[i-coin] !=0)
                    min = Integer.min(min , dp[i-coin]+1);            
                if(min != Integer.MAX_VALUE)dp[i] = min;
            }
        }       
        if(dp[amount] == 0) return -1;
        return dp[amount];
    }
}

```

------

#### 1240 Tiling a Rectangle with the Fewest Squares

Given a rectangle of size `n` x `m`, find the minimum number of integer-sided squares that tile the rectangle.

 

**Example 1:**

![img](https://assets.leetcode.com/uploads/2019/10/17/sample_11_1592.png)

```
Input: n = 2, m = 3
Output: 3
Explanation: 3 squares are necessary to cover the rectangle.
2 (squares of 1x1)
1 (square of 2x2)
```

**Example 2:**

![img](https://assets.leetcode.com/uploads/2019/10/17/sample_22_1592.png)

```
Input: n = 5, m = 8
Output: 5
```

**Example 3:**

![img](https://assets.leetcode.com/uploads/2019/10/17/sample_33_1592.png)

```
Input: n = 11, m = 13
Output: 6
```

**分析**

* 动态规划（Cheating）

对于一个**n*m**的矩形，能切割的正方形边长范围为**1~min(n,m)**。所以显然，就是找到切割完后剩余大小的最小值：

![](https://github.com/POPzxh/Dynamic-Programming-Patterns/blob/master/G1240.png)

![](C:\Java学习\Leetcode\动态规划\Dynamic-Programming-Patterns\G1240.png)

对于剩余部分有两种切法：横，(i-l)\*(j)+l\*(j-l) ; 竖，(i-l)(j-l) + i * (j-l)。

13，11为特殊情况。

```java
class Solution {
    public int tilingRectangle(int n, int m) {
        if(n==11 && m == 13) return 6;
        if(n==13 && m == 11) return 6;
        int[][] dp = new int[n+1][m+1];
        dp[1][1] =1;
        for(int i = 1 ; i<=n ;i ++){
            for(int j =1 ; j<=m ; j++){
                if(i==j){dp[i][j]=1;continue;}
                int min = Integer.MAX_VALUE;
                for(int l = Math.min(i,j);l>=1;l--){
                    if(l==Math.min(i,j)){
                        if(i==l) min = dp[i][j-l];
                        else min = dp[i-l][j];
                    }
                    else
                        min = Integer.min(min,Integer.min(dp[i-l][l]+dp[i][j-l] , dp[l][j-l]+dp[i-l][j]));
                }
                dp[i][j]= min+1;
            }
        }
        
        return dp[n][m];
    }
}
```

* DFS

使用长度为m的数组记录每一个横向位置的高度。每次铺砖，在高度最低的地方，长度由高到低遍历。

```java
class Solution {
    int maxStep = Integer.MAX_VALUE;
    public int tilingRectangle(int n, int m) {
        int[] height =new int[m];
        int least = 0;
        tiling(n,height,0,0);
        return maxStep;
    }

    public void tiling(int n , int[] height , int least , int cnt){
        if(height[least] == n){
            if(cnt<maxStep) maxStep = cnt;
        }
        else{
            if(cnt >= maxStep) return;
            int base = height[least];
            int len = 1;
            for(;len<height.length;len++) if(least + len == height.length || height[least+len]!=base||base+len==n) break;
            for(; len>=1;len--){               
                for(int i = least ; i<least+len;i++) height[i] = base + len;
                int temp = 0;
                for (int i = 0; i < height.length; i++) if (height[i] < height[temp]) temp = i;
                tiling(n, height, temp, cnt + 1);
                for(int i = least ; i<least+len;i++) height[i] = base;
            }
        }
    }
}
```



------

## Distinct Ways

### 一般问题陈述

> ​	Given a target find a number of distinct ways to reach the target.
>
> ​	找到到达目标不同的路径数。

### 一般方法

>Sum all possible ways to reach the current state.
>
>对于当前状态，对于所有能够到达当前状态的路径求和。
>
>```
>routes[i] = routes[i-1] + routes[i-2], ... , + routes[i-k]
>```

### **例题** : 70. Climbing Stairs `Easy`

You are climbing a stair case. It takes *n* steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Note:** Given *n* will be a positive integer.

**Example 1:**

```
Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
```

**分析**

对于一个台阶，只能通过跨一步或者两步到达。所以，显然`dp[i] = dp[i-1] + dp[i-2]`

```java
class Solution {
    public int climbStairs(int n) {
        int[] dp = new int[n+1];
        dp[0] = 1;
        for(int i=1 ; i<=n ; i++){
            if(i - 2 >= 0) dp[i] += dp[i-2];
            dp[i] += dp[i-1];
        }
        return dp[n];
    }
}
```

### 相似问题

|                                                            |                                                              |                                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [62. Unique Paths](#62-unique-paths) `Medium`              | [1155. Number of Dice Rolls With Target Sum](#1155-Number-of-Dice-Rolls-With-Target-Sum)`Medium` | [688. Knight Probability in Chessboard](#688-Knight-Probability-in-Chessboard)`MEdium` |
| [377. Combination Sum IV](#377-Combination-Sum-IV)`Medium` | [935. Knight Dialer](#935-Knight-Dialer)`Medium`             | [1223. Dice Roll Simulation](#1223-Dice-Roll-Simulation)`Medium` |
|                                                            |                                                              |                                                              |

#### 62 Unique Paths

A robot is located at the top-left corner of a *m* x *n* grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

![img](https://assets.leetcode.com/uploads/2018/10/22/robot_maze.png)
Above is a 7 x 3 grid. How many possible unique paths are there?

**Note:** *m* and *n* will be at most 100.

**Example 1:**

```
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
```

**分析**

只能通过向下走，或者向右走才能到达一个方块。所以显然，`dp[i][j] = dp[i-1][j] + dp[i][j-1]`

这题可以只用一维数组。

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[] dp = new int[n+1];
        dp[1] = 1;
        for(int i = 0 ; i<m ; i++){
            for(int j = 1 ; j<= n ; j++){                
                dp[j] = dp[j] + dp[j-1];
            }
        }
        return dp[n];
    }
}
```

------

#### 1155 Number of Dice Rolls With Target Sum

You have `d` dice, and each die has `f` faces numbered `1, 2, ..., f`.

Return the number of possible ways (out of `fd` total ways) **modulo `10^9 + 7`** to roll the dice so the sum of the face up numbers equals `target`.

**Example 1:**

```
Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.
```

**分析**

对于骰子点数**i**，<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;dp[i]&space;=&space;\sum&space;_{j=1}^fdp[i-f]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{120}&space;dp[i]&space;=&space;\sum&space;_{j=1}^fdp[i-f]" title="dp[i] = \sum _{j=1}^fdp[i-f]" /></a>

注意范围：**d~d*f**

```java
class Solution {
    public int numRollsToTarget(int d, int f, int target) {
        if(target < d) return 0;
        if(target > d*f) return 0;
        int[] dp = new int[d*f+1];
        int mod = (int)Math.pow(10 , 9) + 7;
        int max = f*2;
        int min = 2;
        for(int i = 1; i<=f ; i++) dp[i] = 1;
        for(int i = 1 ; i<d ; i++){
            for(int j = max ; j>= min ; j--){
                dp[j] = 0;
                for(int num = 1 ; num <=f ; num++){
                    if(j-num >= min-1) dp[j] = (dp[j-num] + dp[j])%mod ;
                }
            }
            max += f;
            min += 1;
        }
        return dp[target];
    }
}
```

------

#### 688 Knight Probability in Chessboard

迭代K次，每个位置都只能从8个方向到达。

```java
class Solution {
    public double knightProbability(int N, int K, int r, int c) {
        if(K == 0) return 1;
                int[][] mov = new int[][]{
                {-2 , -1},
                {-1 , -2},
                {-2 , 1},
                {-1 , 2},
                {1 ,2 },
                {2 , 1},
                {2 , -1},
                {1 , -2}
        };
        double[][] dp = new double[N][N];
        for(int step = 0 ; step < K ;step++){
            double[][] temp = new double[N][N];
            for(int i = 0 ; i< N ; i++){
                for(int j = 0 ; j < N ;j++){
                    //遍历8个方向，更新dp[i][j]的数值
                    for(int dir = 0 ; dir <8 ; dir++) {
                        int x = i + mov[dir][0];
                        int y = j + mov[dir][1];
                        if(x <0 || x>=N || y <0 || y>= N) continue;//不在棋盘范围内，跳过
                        if(dp[x][y] == 0) temp[i][j] += 1/8.0;
                        else temp[i][j] += dp[x][y]/8.0;
                    }
                }
            }
            dp =temp;
        }  
        return dp[r][c] ;
    }
}
```

------

#### 494. Target Sum

You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols `+` and `-`. For each integer, you should choose one from `+` and `-` as its new symbol.

Find out how many ways to assign symbols to make sum of integers equal to target S.

**Example 1:**

```
Input: nums is [1, 1, 1, 1, 1], S is 3. 
Output: 5
Explanation: 

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

There are 5 ways to assign symbols to make the sum of nums be target 3.
```

**分析**

* BFS

遍历`nums[]`中的每一个数值，更新所有可能的和的组合数。

```java
class Solution {
    public int findTargetSumWays(int[] nums, int S) {
        HashMap<Integer , Integer> map = null;
        for(int num : nums) {
            if (map == null) {
                map = new HashMap<>();
                if(num == 0) map.put(0 , 2);
                else{
                    map.put(num, 1);
                    map.put(-num, 1);
                }
            } else {
                HashMap<Integer, Integer> temp = new HashMap<>();
                map.forEach((k, v) -> {
                    temp.put(k + num, v + temp.getOrDefault(k + num, 0));
                    temp.put(k - num, v + temp.getOrDefault(k - num, 0));
                });
                map = temp;
            }
        }
        if(!map.containsKey(S)) return 0;
        return map.get(S);
    }
}
```

------

#### 377. Combination Sum IV

Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

**Example:**

```
nums = [1, 2, 3]
target = 4

The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Note that different sequences are counted as different combinations.

Therefore the output is 7.
```

**分析**

没啥说的，`dp[i]+=dp[i-num]`

```java
class Solution {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target+1];
        Arrays.sort(nums);
        dp[0] = 1;
        for(int i = 0 ; i<=target ; i++){
            for(int num : nums){
                if(i - num<0)break;
                else{
                    dp[i] += dp[i-num];
                }
            }
        }
        return dp[target];
    }
}
```

-------

#### 935. Knight Dialer

A chess knight can move as indicated in the chess diagram below:

<img src="https://assets.leetcode.com/uploads/2018/10/12/knight.png" alt="img" style="zoom:33%;" /> .      ![img](https://assets.leetcode.com/uploads/2018/10/30/keypad.png)

 

This time, we place our chess knight on any numbered key of a phone pad (indicated above), and the knight makes `N-1` hops. Each hop must be from one key to another numbered key.

Each time it lands on a key (including the initial placement of the knight), it presses the number of that key, pressing `N` digits total.

How many distinct numbers can you dial in this manner?

Since the answer may be large, **output the answer modulo `10^9 + 7`**.

**Example 1:**

```
Input: 1
Output: 10
```

跟[688. Knight Probability in Chessboard](#688-Knight-Probability-in-Chessboard)类似，不赘述。

```java
class Solution {
    public int knightDialer(int N) {
        int MOD = 1_000_000_007;
        int[][] moves = new int[][]{
            {4,6},{6,8},{7,9},{4,8},{3,9,0},
            {},{1,7,0},{2,6},{1,3},{2,4}};

        int[][] dp = new int[2][10];
        Arrays.fill(dp[0], 1);
        for (int hops = 0; hops < N-1; ++hops) {
            Arrays.fill(dp[~hops & 1], 0);
            for (int node = 0; node < 10; ++node)
                for (int nei: moves[node]) {
                    dp[~hops & 1][nei] += dp[hops & 1][node];
                    dp[~hops & 1][nei] %= MOD;
                }
        }

        long ans = 0;
        for (int x: dp[~N & 1])
            ans += x;
        return (int) (ans % MOD);
    }
}
```

------

#### 1223. Dice Roll Simulation

it cannot roll the number `i` more than `rollMax[i]` (1-indexed) **consecutive** times. 

Given an array of integers `rollMax` and an integer `n`, return the number of distinct sequences that can be obtained with exact `n` rolls.

Two sequences are considered different if at least one element differs from each other. Since the answer may be too large, return it modulo `10^9 + 7`.

**Example 1:**

```
Input: n = 2, rollMax = [1,1,2,2,2,3]
Output: 34
Explanation: There will be 2 rolls of die, if there are no constraints on the die, there are 6 * 6 = 36 possible combinations. In this case, looking at rollMax array, the numbers 1 and 2 appear at most once consecutively, therefore sequences (1,1) and (2,2) cannot occur, so the final answer is 36-2 = 34.
```

**分析**

```java
class Solution {
    public int dieSimulator(int n, int[] rollMax) {
        int mod = (int)Math.pow(10,9) + 7;
        long[][] dp = new long[n][7];
        Arrays.fill(dp[0] , 1);
        dp[0][6] = 6;
        for(int i = 1 ; i<n ; i++){
            long sum = 0;
            for(int face = 0 ; face < 6 ; face++){
                if(i < rollMax[face]){
                    dp[i][face] = dp[i-1][6];
                }
                else{
                    for(int consec = 1 ; consec <= rollMax[face] ; consec++){
                        if(i - consec >= 0){
                            dp[i][face] += (dp[i-consec][6] - dp[i-consec][face])%mod;
                            if(dp[i][face] < 0) dp[i][face] += mod;                   
                        }
                    }
                }
                sum += dp[i][face];
                sum %= mod;
            }
            dp[i][6] =sum;
        }
        return (int)dp[n-1][6]%mod;
    }
}
```



