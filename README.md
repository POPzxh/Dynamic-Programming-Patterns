# Dynamic Programming Patterns

<script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
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

#### 相似题目

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

##### 动态规划（Cheating）

对于一个**n*m**的矩形，能切割的正方形边长范围为**1~min(n,m)**。所以显然，就是找到切割完后剩余大小的最小值：

![](https://github.com/POPzxh/Dynamic-Programming-Patterns/blob/master/G1240.png)

------

## Distinct Ways
