<TeXmacs|1.99.5>

<style|<tuple|article|chinese>>

<\body>
  <section|\<#4F7F\>\<#7528\>NDArray\<#6765\>\<#5904\>\<#7406\>\<#6570\>\<#636E\>>

  \<#5BF9\>\<#4E8E\>\<#673A\>\<#5668\>\<#5B66\>\<#4E60\>\<#6765\>\<#8BF4\>\<#FF0C\>\<#5904\>\<#7406\>\<#6570\>\<#636E\>\<#5F80\>\<#5F80\>\<#662F\>\<#4E07\>\<#4E8B\>\<#4E4B\>\<#5F00\>\<#5934\>\<#3002\>\<#5B83\>\<#5305\>\<#542B\>\<#4E24\>\<#4E2A\>\<#90E8\>\<#5206\>\<#FF1A\>\<#6570\>\<#636E\>\<#8BFB\>\<#53D6\>\<#548C\>\<#5F53\>\<#6570\>\<#636E\>\<#5DF2\>\<#7ECF\>\<#5728\>\<#5185\>\<#5B58\>\<#91CC\>\<#65F6\>\<#5982\>\<#4F55\>\<#5904\>\<#7406\>\<#3002\>\<#672C\>\<#7AE0\>\<#5C06\>\<#5173\>\<#6CE8\>\<#540E\>\<#8005\>\<#3002\>\<#6211\>\<#4EEC\>\<#9996\>\<#5148\>\<#4ECB\>\<#7ECD\><code*|NDArray>\<#FF0C\>\<#8FD9\>\<#662F\>MXNet\<#50A8\>\<#5B58\>\<#548C\>\<#53D8\>\<#6362\>\<#6570\>\<#636E\>\<#7684\>\<#4E3B\>\<#8981\>\<#5DE5\>\<#5177\>\<#3002\>\<#5982\>\<#679C\>\<#4F60\>\<#4E4B\>\<#524D\>\<#7528\>\<#8FC7\><code*|NumPy>\<#FF0C\>\<#4F60\>\<#4F1A\>\<#53D1\>\<#73B0\><code*|NDArray>\<#548C\><code*|NumPy>\<#7684\>\<#591A\>\<#7EF4\>\<#6570\>\<#7EC4\>\<#975E\>\<#5E38\>\<#7C7B\>\<#4F3C\>\<#3002\>\<#5F53\>\<#7136\>\<#FF0C\><code*|NDArray>\<#63D0\>\<#4F9B\>\<#66F4\>\<#591A\>\<#7684\>\<#529F\>\<#80FD\>\<#FF0C\>\<#9996\>\<#5148\>\<#662F\>CPU\<#548C\>GPU\<#7684\>\<#5F02\>\<#6B65\>\<#8BA1\>\<#7B97\>\<#FF0C\>\<#5176\>\<#6B21\>\<#662F\>\<#81EA\>\<#52A8\>\<#6C42\>\<#5BFC\>\<#3002\>\<#8FD9\>\<#4E24\>\<#70B9\>\<#4F7F\>\<#5F97\><code*|NDArray>\<#80FD\>\<#66F4\>\<#597D\>\<#5730\>\<#652F\>\<#6301\>\<#673A\>\<#5668\>\<#5B66\>\<#4E60\>\<#3002\>

  <subsection|\<#8BA9\>\<#6211\>\<#4EEC\>\<#5F00\>\<#59CB\>>

  \<#6211\>\<#4EEC\>\<#5148\>\<#4ECB\>\<#7ECD\>\<#6700\>\<#57FA\>\<#672C\>\<#7684\>\<#529F\>\<#80FD\>\<#3002\>\<#5982\>\<#679C\>\<#4F60\>\<#4E0D\>\<#61C2\>\<#6211\>\<#4EEC\>\<#7528\>\<#5230\>\<#7684\>\<#6570\>\<#5B66\>\<#64CD\>\<#4F5C\>\<#4E5F\>\<#4E0D\>\<#7528\>\<#62C5\>\<#5FC3\>\<#FF0C\>\<#4F8B\>\<#5982\>\<#6309\>\<#5143\>\<#7D20\>\<#52A0\>\<#6CD5\>\<#FF0C\>\<#6216\>\<#8005\>\<#6B63\>\<#6001\>\<#5206\>\<#5E03\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#4F1A\>\<#5728\>\<#4E4B\>\<#540E\>\<#7684\>\<#7AE0\>\<#8282\>\<#5206\>\<#522B\>\<#8BE6\>\<#7EC6\>\<#4ECB\>\<#7ECD\>\<#3002\>

  \<#6211\>\<#4EEC\>\<#9996\>\<#5148\>\<#4ECE\><code*|mxnet>\<#5BFC\>\<#5165\><code*|ndarray>\<#8FD9\>\<#4E2A\>\<#5305\>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      from mxnet import ndarray as nd
    </input>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#7136\>\<#540E\>\<#6211\>\<#4EEC\>\<#521B\>\<#5EFA\>\<#4E00\>\<#4E2A\>\<#6709\>3\<#884C\>\<#548C\>4\<#5217\>\<#7684\>2D\<#6570\>\<#7EC4\>\<#FF08\>\<#901A\>\<#5E38\>\<#4E5F\>\<#53EB\>\<#77E9\>\<#9635\>\<#FF09\>\<#FF0C\>\<#5E76\>\<#4E14\>\<#628A\>\<#6BCF\>\<#4E2A\>\<#5143\>\<#7D20\>\<#521D\>\<#59CB\>\<#5316\>\<#6210\>0

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      nd.zeros((3, 4))
    <|unfolded-io>
      [[ 0. \ 0. \ 0. \ 0.]

      \ [ 0. \ 0. \ 0. \ 0.]

      \ [ 0. \ 0. \ 0. \ 0.]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#7C7B\>\<#4F3C\>\<#7684\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#521B\>\<#5EFA\>\<#6570\>\<#7EC4\>\<#6BCF\>\<#4E2A\>\<#5143\>\<#7D20\>\<#88AB\>\<#521D\>\<#59CB\>\<#5316\>\<#6210\>1\<#3002\>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      x = nd.ones((3, 4))
    </input>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      x
    <|unfolded-io>
      [[ 1. \ 1. \ 1. \ 1.]

      \ [ 1. \ 1. \ 1. \ 1.]

      \ [ 1. \ 1. \ 1. \ 1.]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#6216\>\<#8005\>\<#4ECE\>python\<#7684\>\<#6570\>\<#7EC4\>\<#76F4\>\<#63A5\>\<#6784\>\<#9020\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      nd.array([[1, 2], [2, 3]])
    <|unfolded-io>
      [[ 1. \ 2.]

      \ [ 2. \ 3.]]

      \<less\>NDArray 2x2 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#6211\>\<#4EEC\>\<#7ECF\>\<#5E38\>\<#9700\>\<#8981\>\<#521B\>\<#5EFA\>\<#968F\>\<#673A\>\<#6570\>\<#7EC4\>\<#FF0C\>\<#5C31\>\<#662F\>\<#8BF4\>\<#6BCF\>\<#4E2A\>\<#5143\>\<#7D20\>\<#7684\>\<#503C\>\<#90FD\>\<#662F\>\<#968F\>\<#673A\>\<#91C7\>\<#6837\>\<#800C\>\<#6765\>\<#FF0C\>\<#8FD9\>\<#4E2A\>\<#7ECF\>\<#5E38\>\<#88AB\>\<#7528\>\<#6765\>\<#521D\>\<#59CB\>\<#5316\>\<#6A21\>\<#578B\>\<#53C2\>\<#6570\>\<#3002\>\<#4E0B\>\<#9762\>\<#521B\>\<#5EFA\>\<#6570\>\<#7EC4\>\<#FF0C\>\<#5B83\>\<#7684\>\<#5143\>\<#7D20\>\<#670D\>\<#4ECE\>\<#5747\>\<#503C\>0\<#65B9\>\<#5DEE\>1\<#7684\>\<#6B63\>\<#6001\>\<#5206\>\<#5E03\>\<#3002\>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      y = nd.random_normal(0, 1, shape=(3, 4))
    </input>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      y
    <|unfolded-io>
      [[-0.54594457 -1.771029 \ \ -2.35562968 -0.45138445]

      \ [ 0.54144019 \ 0.57938355 \ 2.67850661 -1.85608196]

      \ [ 1.25463438 -1.9768796 \ -0.54877394 -0.20801921]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#8DDF\><code*|NumPy>\<#4E00\>\<#6837\>\<#FF0C\>\<#6BCF\>\<#4E2A\>\<#6570\>\<#7EC4\>\<#7684\>\<#5F62\>\<#72B6\>\<#53EF\>\<#4EE5\>\<#901A\>\<#8FC7\><code*|.shape>\<#6765\>\<#83B7\>\<#53D6\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      y.shape
    <|unfolded-io>
      (3, 4)
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#5B83\>\<#7684\>\<#5927\>\<#5C0F\>\<#FF0C\>\<#5C31\>\<#662F\>\<#603B\>\<#5143\>\<#7D20\>\<#4E2A\>\<#6570\>\<#FF0C\>\<#662F\>\<#5F62\>\<#72B6\>\<#7684\>\<#7D2F\>\<#4E58\>\<#3002\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      y.size
    <|unfolded-io>
      12
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  <subsection|\<#64CD\>\<#4F5C\>\<#7B26\>>

  NDArray\<#652F\>\<#6301\>\<#5927\>\<#91CF\>\<#7684\>\<#6570\>\<#5B66\>\<#64CD\>\<#4F5C\>\<#7B26\>\<#FF0C\>\<#4F8B\>\<#5982\>\<#6309\>\<#5143\>\<#7D20\>\<#52A0\>\<#6CD5\>\<#FF1A\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      x + y
    <|unfolded-io>
      [[ 0.45405543 -0.771029 \ \ -1.35562968 \ 0.54861557]

      \ [ 1.54144025 \ 1.57938361 \ 3.67850661 -0.85608196]

      \ [ 2.25463438 -0.9768796 \ \ 0.45122606 \ 0.7919808 ]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#4E58\>\<#6CD5\>\<#FF1A\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      x * y
    <|unfolded-io>
      [[-0.54594457 -1.771029 \ \ -2.35562968 -0.45138445]

      \ [ 0.54144019 \ 0.57938355 \ 2.67850661 -1.85608196]

      \ [ 1.25463438 -1.9768796 \ -0.54877394 -0.20801921]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#6307\>\<#6570\>\<#8FD0\>\<#7B97\>\<#FF1A\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      nd.exp(y)
    <|unfolded-io>
      [[ \ 0.57929432 \ \ 0.17015781 \ \ 0.09483377 \ \ 0.63674599]

      \ [ \ 1.71847999 \ \ 1.78493774 \ 14.56332874 \ \ 0.15628375]

      \ [ \ 3.50655603 \ \ 0.13850074 \ \ 0.57765764 \ \ 0.81219143]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#4E5F\>\<#53EF\>\<#4EE5\>\<#8F6C\>\<#7F6E\>\<#4E00\>\<#4E2A\>\<#77E9\>\<#9635\>\<#7136\>\<#540E\>\<#8BA1\>\<#7B97\>\<#77E9\>\<#9635\>\<#4E58\>\<#6CD5\>:

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      nd.dot(x, y.T)
    <|unfolded-io>
      [[-5.12398815 \ 1.94324839 -1.47903836]

      \ [-5.12398815 \ 1.94324839 -1.47903836]

      \ [-5.12398815 \ 1.94324827 -1.47903848]]

      \<less\>NDArray 3x3 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  <subsection|\<#5E7F\>\<#64AD\>>

  \<#5F53\>\<#4E8C\>\<#5143\>\<#64CD\>\<#4F5C\>\<#7B26\>\<#5DE6\>\<#53F3\>\<#4E24\>\<#8FB9\>ndarray\<#5F62\>\<#72B6\>\<#4E0D\>\<#4E00\>\<#6837\>\<#65F6\>\<#FF0C\>\<#7CFB\>\<#7EDF\>\<#4F1A\>\<#5C1D\>\<#8BD5\>\<#5C06\>\<#5176\>\<#590D\>\<#5236\>\<#5230\>\<#4E00\>\<#4E2A\>\<#5171\>\<#540C\>\<#7684\>\<#5F62\>\<#72B6\>\<#3002\>\<#4F8B\>\<#5982\><code*|a>\<#7684\>\<#7B2C\>0\<#7EF4\>\<#662F\>3,
  <code*|b>\<#7684\>\<#7B2C\>0\<#7EF4\>\<#662F\>1\<#FF0C\>\<#90A3\>\<#4E48\><code*|a+b>\<#65F6\>\<#4F1A\>\<#5C06\><code*|b>\<#6CBF\>\<#7740\>\<#7B2C\>0\<#7EF4\>\<#590D\>\<#5236\>3\<#904D\>\<#FF1A\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      a = nd.arange(3).reshape((3,1))

      b = nd.arange(2).reshape((1,2))

      print('a:', a)

      print('b:', b)

      print('a+b:', a+b)
    <|unfolded-io>
      a:\ 

      [[ 0.]

      \ [ 1.]

      \ [ 2.]]

      \<less\>NDArray 3x1 @cpu(0)\<gtr\>

      b:\ 

      [[ 0. \ 1.]]

      \<less\>NDArray 1x2 @cpu(0)\<gtr\>

      a+b:\ 

      [[ 0. \ 1.]

      \ [ 1. \ 2.]

      \ [ 2. \ 3.]]

      \<less\>NDArray 3x2 @cpu(0)\<gtr\>
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  <subsection|\<#8DDF\>NumPy\<#7684\>\<#8F6C\>\<#6362\>>

  ndarray\<#53EF\>\<#4EE5\>\<#5F88\>\<#65B9\>\<#4FBF\>\<#540C\>numpy\<#8FDB\>\<#884C\>\<#8F6C\>\<#6362\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      import numpy as np

      x = np.ones((2,3))

      y = nd.array(x) \ # numpy -\<gtr\> mxnet

      z = y.asnumpy() \ # mxnet -\<gtr\> numpy

      print([z, y])
    <|unfolded-io>
      [array([[ 1., \ 1., \ 1.],

      \ \ \ \ \ \ \ [ 1., \ 1., \ 1.]], dtype=float32),\ 

      [[ 1. \ 1. \ 1.]

      \ [ 1. \ 1. \ 1.]]

      \<less\>NDArray 2x3 @cpu(0)\<gtr\>]
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  <subsection|\<#66FF\>\<#6362\>\<#64CD\>\<#4F5C\>>

  \<#5728\>\<#524D\>\<#9762\>\<#7684\>\<#6837\>\<#4F8B\>\<#4E2D\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#4E3A\>\<#6BCF\>\<#4E2A\>\<#64CD\>\<#4F5C\>\<#65B0\>\<#5F00\>\<#5185\>\<#5B58\>\<#6765\>\<#5B58\>\<#50A8\>\<#5B83\>\<#7684\>\<#7ED3\>\<#679C\>\<#3002\>\<#4F8B\>\<#5982\>\<#FF0C\>\<#5982\>\<#679C\>\<#6211\>\<#4EEC\>\<#5199\><math|y=x+y>,
  \<#6211\>\<#4EEC\>\<#4F1A\>\<#628A\><code*|y>\<#4ECE\>\<#73B0\>\<#5728\>\<#6307\>\<#5411\>\<#7684\>\<#5B9E\>\<#4F8B\>\<#8F6C\>\<#5230\>\<#65B0\>\<#5EFA\>\<#7684\>\<#5B9E\>\<#4F8B\>\<#4E0A\>\<#53BB\>\<#3002\>\<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#7528\>Python\<#7684\><code*|id()>\<#51FD\>\<#6570\>\<#6765\>\<#770B\>\<#8FD9\>\<#4E2A\>\<#662F\>\<#600E\>\<#4E48\>\<#6267\>\<#884C\>\<#7684\>\<#FF1A\>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      x = nd.ones((3, 4))

      y = nd.ones((3, 4))

      before = id(y)

      y = y + x
    </input>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      id(y) == before
    <|unfolded-io>
      False
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  <code|False>

  \<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#628A\>\<#7ED3\>\<#679C\>\<#901A\>\<#8FC7\><code*|[:]>\<#5199\>\<#5230\>\<#4E00\>\<#4E2A\>\<#4E4B\>\<#524D\>\<#5F00\>\<#597D\>\<#7684\>\<#6570\>\<#7EC4\>\<#91CC\>\<#FF1A\>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      z = nd.zeros_like(x)

      before = id(z)

      z[:] = x + y
    </input>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      id(z) == before
    <|unfolded-io>
      True
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#4F46\>\<#662F\>\<#8FD9\>\<#91CC\>\<#6211\>\<#4EEC\>\<#8FD8\>\<#662F\>\<#4E3A\><code*|x+y>\<#521B\>\<#5EFA\>\<#4E86\>\<#4E34\>\<#65F6\>\<#7A7A\>\<#95F4\>\<#FF0C\>\<#7136\>\<#540E\>\<#518D\>\<#590D\>\<#5236\>\<#5230\><code*|z>\<#3002\>\<#9700\>\<#8981\>\<#907F\>\<#514D\>\<#8FD9\>\<#4E2A\>\<#5F00\>\<#9500\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#4F7F\>\<#7528\>\<#64CD\>\<#4F5C\>\<#7B26\>\<#7684\>\<#5168\>\<#540D\>\<#7248\>\<#672C\>\<#4E2D\>\<#7684\><code*|out>\<#53C2\>\<#6570\>\<#FF1A\>

  <\session|python|default>
    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      nd.elemwise_add(x, y, out=z)
    <|unfolded-io>
      [[ 3. \ 3. \ 3. \ 3.]

      \ [ 3. \ 3. \ 3. \ 3.]

      \ [ 3. \ 3. \ 3. \ 3.]]

      \<less\>NDArray 3x4 @cpu(0)\<gtr\>
    </unfolded-io>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      id(z) == before
    <|unfolded-io>
      True
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  \<#5982\>\<#679C\>\<#53EF\>\<#4EE5\>\<#73B0\>\<#6709\>\<#7684\>\<#6570\>\<#7EC4\>\<#4E4B\>\<#540E\>\<#4E0D\>\<#4F1A\>\<#518D\>\<#7528\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#4E5F\>\<#53EF\>\<#4EE5\>\<#7528\>\<#590D\>\<#5236\>\<#64CD\>\<#4F5C\>\<#7B26\>\<#8FBE\>\<#5230\>\<#8FD9\>\<#4E2A\>\<#76EE\>\<#7684\>\<#FF1A\>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      before = id(x)

      x += y
    </input>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      id(x) == before
    <|unfolded-io>
      True
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>

  <subsection|\<#603B\>\<#7ED3\>>

  ndarray\<#6A21\>\<#5757\>\<#63D0\>\<#4F9B\>\<#4E00\>\<#7CFB\>\<#5217\>\<#591A\>\<#7EF4\>\<#6570\>\<#7EC4\>\<#64CD\>\<#4F5C\>\<#51FD\>\<#6570\>\<#3002\>\<#6240\>\<#6709\>\<#51FD\>\<#6570\>\<#5217\>\<#8868\>\<#53EF\>\<#4EE5\>\<#53C2\>\<#89C1\><hlink|NDArray
  API\<#6587\>\<#6863\>|https://mxnet.incubator.apache.org/api/python/ndarray.html>\<#3002\>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
    <associate|auto-4|<tuple|1.3|?>>
    <associate|auto-5|<tuple|1.4|?>>
    <associate|auto-6|<tuple|1.5|?>>
    <associate|auto-7|<tuple|1.6|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>\<#4F7F\>\<#7528\>NDArray\<#6765\>\<#5904\>\<#7406\>\<#6570\>\<#636E\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>\<#8BA9\>\<#6211\>\<#4EEC\>\<#5F00\>\<#59CB\>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>\<#64CD\>\<#4F5C\>\<#7B26\>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1tab>|1.3<space|2spc>\<#5E7F\>\<#64AD\>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1tab>|1.4<space|2spc>\<#8DDF\>NumPy\<#7684\>\<#8F6C\>\<#6362\>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|1.5<space|2spc>\<#66FF\>\<#6362\>\<#64CD\>\<#4F5C\>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|1.6<space|2spc>\<#603B\>\<#7ED3\>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>