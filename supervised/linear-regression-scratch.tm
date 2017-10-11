<TeXmacs|1.99.5>

<style|<tuple|article|literate|chinese>>

<\body>
  <chapter|\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\><math|>:\<#4ECE\>0\<#5F00\>\<#59CB\><hlink||https://zh.gluon.ai/linear-regression-scratch.html#线性回归-----从0开始>>

  \<#867D\>\<#7136\>\<#5F3A\>\<#5927\>\<#7684\>\<#6DF1\>\<#5EA6\>\<#5B66\>\<#4E60\>\<#6846\>\<#67B6\>\<#53EF\>\<#4EE5\>\<#51CF\>\<#5C11\>\<#5F88\>\<#591A\>\<#91CD\>\<#590D\>\<#6027\>\<#5DE5\>\<#4F5C\>\<#FF0C\>\<#4F46\>\<#5982\>\<#679C\>\<#4F60\>\<#8FC7\>\<#4E8E\>\<#4F9D\>\<#8D56\>\<#5B83\>\<#63D0\>\<#4F9B\>\<#7684\>\<#4FBF\>\<#5229\>\<#62BD\>\<#8C61\>\<#FF0C\>\<#90A3\>\<#4E48\>\<#4F60\>\<#53EF\>\<#80FD\>\<#4E0D\>\<#4F1A\>\<#5F88\>\<#5BB9\>\<#6613\>\<#5730\>\<#7406\>\<#89E3\>\<#5230\>\<#5E95\>\<#6DF1\>\<#5EA6\>\<#5B66\>\<#4E60\>\<#662F\>\<#5982\>\<#4F55\>\<#5DE5\>\<#4F5C\>\<#7684\>\<#3002\>\<#6240\>\<#4EE5\>\<#6211\>\<#4EEC\>\<#7684\>\<#7B2C\>\<#4E00\>\<#4E2A\>\<#6559\>\<#7A0B\>\<#662F\>\<#5982\>\<#4F55\>\<#53EA\>\<#5229\>\<#7528\>ndarray\<#548C\>autograd\<#6765\>\<#5B9E\>\<#73B0\>\<#4E00\>\<#4E2A\>\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>\<#7684\>\<#8BAD\>\<#7EC3\>\<#3002\>

  <section|\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>>

  \<#7ED9\>\<#5B9A\>\<#4E00\>\<#4E2A\>\<#6570\>\<#636E\>\<#70B9\>\<#96C6\>\<#5408\><code*|X>\<#548C\>\<#5BF9\>\<#5E94\>\<#7684\>\<#76EE\>\<#6807\>\<#503C\><code*|y>\<#FF0C\>\<#7EBF\>\<#6027\>\<#6A21\>\<#578B\>\<#7684\>\<#76EE\>\<#6807\>\<#662F\>\<#627E\>\<#4E00\>\<#6839\>\<#7EBF\>\<#FF0C\>\<#5176\>\<#7531\>\<#5411\>\<#91CF\><math|\<b-w\>>\<#548C\>\<#4F4D\>\<#79FB\><code*|b>\<#7EC4\>\<#6210\>\<#FF0C\>\<#6765\>\<#6700\>\<#597D\>\<#5730\>\<#8FD1\>\<#4F3C\>\<#6BCF\>\<#4E2A\>\<#6837\>\<#672C\><math|X<around*|[|i|]>>\<#548C\><math|y<around*|[|i|]>>\<#3002\>\<#7528\>\<#6570\>\<#5B66\>\<#7B26\>\<#53F7\>\<#6765\>\<#8868\>\<#793A\>\<#5C31\>\<#662F\>\<#6211\>\<#4EEC\>\<#5C06\>\<#5B66\><math|\<b-w\>>\<#548C\><code*|b>\<#6765\>\<#9884\>\<#6D4B\>

  <\equation*>
    <wide|y|^>=X\<b-w\>+b
  </equation*>

  \<#5E76\>\<#6700\>\<#5C0F\>\<#5316\>\<#6240\>\<#6709\>\<#6570\>\<#636E\>\<#70B9\>\<#4E0A\>\<#7684\>\<#5E73\>\<#65B9\>\<#8BEF\>\<#5DEE\>

  <label|MathJax-Element-2-Frame><label|MathJax-Span-13><label|MathJax-Span-14><label|MathJax-Span-15><label|MathJax-Span-16>\<big-sum\><label|MathJax-Span-17><label|MathJax-Span-18><label|MathJax-Span-19>i<label|MathJax-Span-20>=<label|MathJax-Span-21>1<label|MathJax-Span-22>n<label|MathJax-Span-23>(<label|MathJax-Span-24><label|MathJax-Span-25><label|MathJax-Span-26><label|MathJax-Span-27><label|MathJax-Span-28>y<label|MathJax-Span-29>\<#302\><nbsp><label|MathJax-Span-30>i<label|MathJax-Span-31>\<minus\><label|MathJax-Span-32><label|MathJax-Span-33>y<label|MathJax-Span-34>i<label|MathJax-Span-35><label|MathJax-Span-36>)<label|MathJax-Span-37>2<label|MathJax-Span-38>.<math|<above|<below|\<big-sum\>|i=1>|n><around*|(|<above|y|^><rsub|i>\<minus\>y<rsub|i>|)><rsup|2>.>

  \<#4F60\>\<#53EF\>\<#80FD\>\<#4F1A\>\<#5BF9\>\<#6211\>\<#4EEC\>\<#628A\>\<#53E4\>\<#8001\>\<#7684\>\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>\<#4F5C\>\<#4E3A\>\<#6DF1\>\<#5EA6\>\<#5B66\>\<#4E60\>\<#7684\>\<#4E00\>\<#4E2A\>\<#6837\>\<#4F8B\>\<#8868\>\<#793A\>\<#5F88\>\<#5947\>\<#602A\>\<#3002\>\<#5B9E\>\<#9645\>\<#4E0A\>\<#7EBF\>\<#6027\>\<#6A21\>\<#578B\>\<#662F\>\<#6700\>\<#7B80\>\<#5355\>\<#4F46\>\<#4E5F\>\<#53EF\>\<#80FD\>\<#662F\>\<#6700\>\<#6709\>\<#7528\>\<#7684\>\<#795E\>\<#7ECF\>\<#7F51\>\<#7EDC\>\<#3002\>\<#4E00\>\<#4E2A\>\<#795E\>\<#7ECF\>\<#7F51\>\<#7EDC\>\<#5C31\>\<#662F\>\<#4E00\>\<#4E2A\>\<#7531\>\<#8282\>\<#70B9\>\<#FF08\>\<#795E\>\<#7ECF\>\<#5143\>\<#FF09\>\<#548C\>\<#6709\>\<#5411\>\<#8FB9\>\<#7EC4\>\<#6210\>\<#7684\>\<#96C6\>\<#5408\>\<#3002\>\<#6211\>\<#4EEC\>\<#4E00\>\<#822C\>\<#628A\>\<#4E00\>\<#4E9B\>\<#8282\>\<#70B9\>\<#7EC4\>\<#6210\>\<#5C42\>\<#FF0C\>\<#6BCF\>\<#4E00\>\<#5C42\>\<#4F7F\>\<#7528\>\<#4E0B\>\<#4E00\>\<#5C42\>\<#7684\>\<#8282\>\<#70B9\>\<#4F5C\>\<#4E3A\>\<#8F93\>\<#5165\>\<#FF0C\>\<#5E76\>\<#8F93\>\<#51FA\>\<#7ED9\>\<#4E0A\>\<#9762\>\<#5C42\>\<#4F7F\>\<#7528\>\<#3002\>\<#4E3A\>\<#4E86\>\<#8BA1\>\<#7B97\>\<#4E00\>\<#4E2A\>\<#8282\>\<#70B9\>\<#503C\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#5C06\>\<#8F93\>\<#5165\>\<#8282\>\<#70B9\>\<#503C\>\<#505A\>\<#52A0\>\<#6743\>\<#548C\>\<#FF0C\>\<#7136\>\<#540E\>\<#518D\>\<#52A0\>\<#4E0A\>\<#4E00\>\<#4E2A\>\<#6FC0\>\<#6D3B\>\<#51FD\>\<#6570\>\<#3002\>\<#5BF9\>\<#4E8E\>\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>\<#800C\>\<#8A00\>\<#FF0C\>\<#5B83\>\<#662F\>\<#4E00\>\<#4E2A\>\<#4E24\>\<#5C42\>\<#795E\>\<#7ECF\>\<#7F51\>\<#7EDC\>\<#FF0C\>\<#5176\>\<#4E2D\>\<#7B2C\>\<#4E00\>\<#5C42\>\<#662F\>\<#FF08\>\<#4E0B\>\<#56FE\>\<#6A59\>\<#8272\>\<#70B9\>\<#FF09\>\<#8F93\>\<#5165\>\<#FF0C\>\<#6BCF\>\<#4E2A\>\<#8282\>\<#70B9\>\<#5BF9\>\<#5E94\>\<#8F93\>\<#5165\>\<#6570\>\<#636E\>\<#70B9\>\<#7684\>\<#4E00\>\<#4E2A\>\<#7EF4\>\<#5EA6\>\<#FF0C\>\<#7B2C\>\<#4E8C\>\<#5C42\>\<#662F\>\<#5355\>\<#8F93\>\<#51FA\>\<#8282\>\<#70B9\>\<#FF08\>\<#4E0B\>\<#56FE\>\<#7EFF\>\<#8272\>\<#70B9\>\<#FF09\>\<#FF0C\>\<#5B83\>\<#4F7F\>\<#7528\>\<#8EAB\>\<#4EFD\>\<#51FD\>\<#6570\>\<#FF08\><label|MathJax-Element-3-Frame><label|MathJax-Span-39><label|MathJax-Span-40><label|MathJax-Span-41>f<label|MathJax-Span-42>(<label|MathJax-Span-43>x<label|MathJax-Span-44>)<label|MathJax-Span-45>=<label|MathJax-Span-46>x<math|f<around*|(|x|)>=x>\<#FF09\>\<#4F5C\>\<#4E3A\>\<#6FC0\>\<#6D3B\>\<#51FD\>\<#6570\>\<#3002\>

  <image|https://zh.gluon.ai/_images/simple-net-linear.png|0.5w|||>

  <section|\<#521B\>\<#5EFA\>\<#6570\>\<#636E\>\<#96C6\>><label|创建数据集>

  \<#8FD9\>\<#91CC\>\<#6211\>\<#4EEC\>\<#4F7F\>\<#7528\>\<#4E00\>\<#4E2A\>\<#4EBA\>\<#5DE5\>\<#6570\>\<#636E\>\<#96C6\>\<#6765\>\<#628A\>\<#4E8B\>\<#60C5\>\<#5F04\>\<#7B80\>\<#5355\>\<#4E9B\>\<#FF0C\>\<#56E0\>\<#4E3A\>\<#8FD9\>\<#6837\>\<#6211\>\<#4EEC\>\<#5C06\>\<#77E5\>\<#9053\>\<#771F\>\<#5B9E\>\<#7684\>\<#6A21\>\<#578B\>\<#662F\>\<#4EC0\>\<#4E48\>\<#6837\>\<#7684\>\<#3002\>\<#5177\>\<#4F53\>\<#6765\>\<#8BF4\>\<#6211\>\<#4EEC\>\<#4F7F\>\<#7528\>\<#5982\>\<#4E0B\>\<#65B9\>\<#6CD5\>\<#6765\>\<#751F\>\<#6210\>\<#6570\>\<#636E\>

  <code*|y[i]<nbsp>=<nbsp>2<nbsp>*<nbsp>X[i][0]<nbsp>-<nbsp>3.4<nbsp>*<nbsp>X[i][1]<nbsp>+<nbsp>4.2<nbsp>+<nbsp>noise>

  \<#8FD9\>\<#91CC\>\<#566A\>\<#97F3\>\<#670D\>\<#4ECE\>\<#5747\>\<#503C\>0\<#548C\>\<#6807\>\<#51C6\>\<#5DEE\>\<#4E3A\>0.01\<#7684\>\<#6B63\>\<#6001\>\<#5206\>\<#5E03\>\<#3002\>

  <code|In [1]:>

  <\code>
    from mxnet import ndarray as nd

    \;

    \;

    from mxnet import autograd

    \;

    \;

    \;

    num_inputs = 2

    \;

    \;

    num_examples = 1000

    \;

    \;

    \;

    true_w = [2, -3.4]

    \;

    \;

    true_b = 4.2

    \;

    \;

    \;

    X = nd.random_normal(shape=(num_examples, num_inputs))

    \;

    \;

    y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b

    \;

    \;

    y += .01 * nd.random_normal(shape=y.shape)
  </code>

  \<#6CE8\>\<#610F\>\<#5230\><code*|X>\<#7684\>\<#6BCF\>\<#4E00\>\<#884C\>\<#662F\>\<#4E00\>\<#4E2A\>\<#957F\>\<#5EA6\>\<#4E3A\>2\<#7684\>\<#5411\>\<#91CF\>\<#FF0C\>\<#800C\><code*|y>\<#7684\>\<#6BCF\>\<#4E00\>\<#884C\>\<#662F\>\<#4E00\>\<#4E2A\>\<#957F\>\<#5EA6\>\<#4E3A\>1\<#7684\>\<#5411\>\<#91CF\>\<#FF08\>\<#6807\>\<#91CF\>\<#FF09\>\<#3002\>

  <code|In [2]:>

  <code|print(X[0], y[0])>

  \;

  <\code>
    [ 2.21220636 \ 1.16307867]

    \<less\>NDArray 2 @cpu(0)\<gtr\>

    [ 4.6620779]

    \<less\>NDArray 1 @cpu(0)\<gtr\>
  </code>

  <section|\<#6570\>\<#636E\>\<#8BFB\>\<#53D6\>><label|数据读取>

  \<#5F53\>\<#6211\>\<#4EEC\>\<#5F00\>\<#59CB\>\<#8BAD\>\<#7EC3\>\<#795E\>\<#7ECF\>\<#7F51\>\<#7EDC\>\<#7684\>\<#65F6\>\<#5019\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#9700\>\<#8981\>\<#4E0D\>\<#65AD\>\<#8BFB\>\<#53D6\>\<#6570\>\<#636E\>\<#5757\>\<#3002\>\<#8FD9\>\<#91CC\>\<#6211\>\<#4EEC\>\<#5B9A\>\<#4E49\>\<#4E00\>\<#4E2A\>\<#51FD\>\<#6570\>\<#5B83\>\<#6BCF\>\<#6B21\>\<#8FD4\>\<#56DE\><code*|batch_size>\<#4E2A\>\<#968F\>\<#673A\>\<#7684\>\<#6837\>\<#672C\>\<#548C\>\<#5BF9\>\<#5E94\>\<#7684\>\<#76EE\>\<#6807\>\<#3002\>\<#6211\>\<#4EEC\>\<#901A\>\<#8FC7\>python\<#7684\><code*|yield>\<#6765\>\<#6784\>\<#9020\>\<#4E00\>\<#4E2A\>\<#8FED\>\<#4EE3\>\<#5668\>\<#3002\>

  <code|In [3]:>

  <\code>
    import random

    \;

    \;

    batch_size = 10

    \;

    \;

    def data_iter():

    \;

    \ \ \ \ 

    # \<#4EA7\>\<#751F\>\<#4E00\>\<#4E2A\>\<#968F\>\<#673A\>\<#7D22\>\<#5F15\>

    \;

    \ \ \ \ 

    idx = list(range(num_examples))

    \;

    \ \ \ \ 

    random.shuffle(idx)

    \;

    \ \ \ \ 

    for i in range(0, num_examples, batch_size):

    \;

    \ \ \ \ \ \ \ \ 

    j = nd.array(idx[i:min(i+batch_size,num_examples)])

    \;

    \ \ \ \ \ \ \ \ 

    yield nd.take(X, j), nd.take(y, j)
  </code>

  \<#4E0B\>\<#9762\>\<#4EE3\>\<#7801\>\<#8BFB\>\<#53D6\>\<#7B2C\>\<#4E00\>\<#4E2A\>\<#968F\>\<#673A\>\<#6570\>\<#636E\>\<#5757\>

  <code|In [4]:>

  <\code>
    for data, label in data_iter():

    \;

    \ \ \ \ 

    print(data, label)

    \;

    \ \ \ \ 

    break
  </code>

  \;

  <\code>
    [[ 0.74177277 \ 0.85860491]

    \ [-0.47508195 -0.24207895]

    \ [-1.07131875 \ 0.9324615 ]

    \ [-0.55172437 -0.27302608]

    \ [-2.25770831 -0.1240116 ]

    \ [-1.88908994 -2.51524496]

    \ [ 1.02409828 \ 0.56759083]

    \ [ 0.46068263 -1.04109049]

    \ [ 0.2444218 \ -0.68106437]

    \ [-1.03488326 \ 0.11296888]]

    \<less\>NDArray 10x2 @cpu(0)\<gtr\>

    [ 2.77818441 \ 4.04588223 -1.11165142 \ 4.02571392 \ 0.11009595
    \ 8.95327759

    \ \ 4.32130575 \ 8.67033005 \ 7.01202297 \ 1.73317111]

    \<less\>NDArray 10 @cpu(0)\<gtr\>
  </code>

  <section|\<#521D\>\<#59CB\>\<#5316\>\<#6A21\>\<#578B\>\<#53C2\>\<#6570\>><label|初始化模型参数>

  \<#4E0B\>\<#9762\>\<#6211\>\<#4EEC\>\<#968F\>\<#673A\>\<#521D\>\<#59CB\>\<#5316\>\<#6A21\>\<#578B\>\<#53C2\>\<#6570\>

  <code|In [5]:>

  <\code>
    w = nd.random_normal(shape=(num_inputs, 1))

    \;

    \;

    b = nd.zeros((1,))

    \;

    \;

    params = [w, b]
  </code>

  \<#4E4B\>\<#540E\>\<#8BAD\>\<#7EC3\>\<#65F6\>\<#6211\>\<#4EEC\>\<#9700\>\<#8981\>\<#5BF9\>\<#8FD9\>\<#4E9B\>\<#53C2\>\<#6570\>\<#6C42\>\<#5BFC\>\<#6765\>\<#66F4\>\<#65B0\>\<#5B83\>\<#4EEC\>\<#7684\>\<#503C\>\<#FF0C\>\<#6240\>\<#4EE5\>\<#6211\>\<#4EEC\>\<#9700\>\<#8981\>\<#521B\>\<#5EFA\>\<#5B83\>\<#4EEC\>\<#7684\>\<#68AF\>\<#5EA6\>\<#3002\>

  <code|In [6]:>

  <\code>
    for param in params:

    \;

    \ \ \ \ 

    param.attach_grad()
  </code>

  <section|\<#5B9A\>\<#4E49\>\<#6A21\>\<#578B\>><label|定义模型>

  \<#7EBF\>\<#6027\>\<#6A21\>\<#578B\>\<#5C31\>\<#662F\>\<#5C06\>\<#8F93\>\<#5165\>\<#548C\>\<#6A21\>\<#578B\>\<#505A\>\<#4E58\>\<#6CD5\>\<#518D\>\<#52A0\>\<#4E0A\>\<#504F\>\<#79FB\>\<#FF1A\>

  <code|In [7]:>

  <\code>
    def net(X):

    \;

    \ \ \ \ 

    return nd.dot(X, w) + b
  </code>

  <section|\<#635F\>\<#5931\>\<#51FD\>\<#6570\>><label|损失函数>

  \<#6211\>\<#4EEC\>\<#4F7F\>\<#7528\>\<#5E38\>\<#89C1\>\<#7684\>\<#5E73\>\<#65B9\>\<#8BEF\>\<#5DEE\>\<#6765\>\<#8861\>\<#91CF\>\<#9884\>\<#6D4B\>\<#76EE\>\<#6807\>\<#548C\>\<#771F\>\<#5B9E\>\<#76EE\>\<#6807\>\<#4E4B\>\<#95F4\>\<#7684\>\<#5DEE\>\<#8DDD\>\<#3002\>

  <code|In [8]:>

  <\code>
    def square_loss(yhat, y):

    \;

    \ \ \ \ 

    # \<#6CE8\>\<#610F\>\<#8FD9\>\<#91CC\>\<#6211\>\<#4EEC\>\<#628A\>y\<#53D8\>\<#5F62\>\<#6210\>yhat\<#7684\>\<#5F62\>\<#72B6\>\<#6765\>\<#907F\>\<#514D\>\<#81EA\>\<#52A8\>\<#5E7F\>\<#64AD\>

    \;

    \ \ \ \ 

    return (yhat - y.reshape(yhat.shape)) ** 2
  </code>

  <section|\<#4F18\>\<#5316\>><label|优化>

  \<#867D\>\<#7136\>\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>\<#6709\>\<#663E\>\<#8BD5\>\<#89E3\>\<#FF0C\>\<#4F46\>\<#7EDD\>\<#5927\>\<#90E8\>\<#5206\>\<#6A21\>\<#578B\>\<#5E76\>\<#6CA1\>\<#6709\>\<#3002\>\<#6240\>\<#4EE5\>\<#6211\>\<#4EEC\>\<#8FD9\>\<#91CC\>\<#901A\>\<#8FC7\>\<#968F\>\<#673A\>\<#68AF\>\<#5EA6\>\<#4E0B\>\<#964D\>\<#6765\>\<#6C42\>\<#89E3\>\<#3002\>\<#6BCF\>\<#4E00\>\<#6B65\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#5C06\>\<#6A21\>\<#578B\>\<#53C2\>\<#6570\>\<#6CBF\>\<#7740\>\<#68AF\>\<#5EA6\>\<#7684\>\<#53CD\>\<#65B9\>\<#5411\>\<#8D70\>\<#7279\>\<#5B9A\>\<#8DDD\>\<#79BB\>\<#FF0C\>\<#8FD9\>\<#4E2A\>\<#8DDD\>\<#79BB\>\<#4E00\>\<#822C\>\<#53EB\>\<#5B66\>\<#4E60\>\<#7387\>\<#3002\>\<#FF08\>\<#6211\>\<#4EEC\>\<#4F1A\>\<#4E4B\>\<#540E\>\<#4E00\>\<#76F4\>\<#4F7F\>\<#7528\>\<#8FD9\>\<#4E2A\>\<#51FD\>\<#6570\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#5C06\>\<#5176\>\<#4FDD\>\<#5B58\>\<#5728\><hlink|utils.py|https://zh.gluon.ai/utils.py>\<#3002\>\<#FF09\>

  <code|In [9]:>

  <\code>
    def SGD(params, lr):

    \;

    \ \ \ \ 

    for param in params:

    \;

    \ \ \ \ \ \ \ \ 

    param[:] = param - lr * param.grad
  </code>

  <section|\<#8BAD\>\<#7EC3\>><label|训练>

  \<#73B0\>\<#5728\>\<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#5F00\>\<#59CB\>\<#8BAD\>\<#7EC3\>\<#4E86\>\<#3002\>\<#8BAD\>\<#7EC3\>\<#901A\>\<#5E38\>\<#9700\>\<#8981\>\<#8FED\>\<#4EE3\>\<#6570\>\<#636E\>\<#6570\>\<#6B21\>\<#FF0C\>\<#4E00\>\<#6B21\>\<#8FED\>\<#4EE3\>\<#91CC\>\<#FF0C\>\<#6211\>\<#4EEC\>\<#6BCF\>\<#6B21\>\<#968F\>\<#673A\>\<#8BFB\>\<#53D6\>\<#56FA\>\<#5B9A\>\<#6570\>\<#4E2A\>\<#6570\>\<#636E\>\<#70B9\>\<#FF0C\>\<#8BA1\>\<#7B97\>\<#68AF\>\<#5EA6\>\<#5E76\>\<#66F4\>\<#65B0\>\<#6A21\>\<#578B\>\<#53C2\>\<#6570\>\<#3002\>

  <code|In [10]:>

  <\code>
    epochs = 5

    \;

    \;

    learning_rate = .001

    \;

    \;

    for e in range(epochs):

    \;

    \ \ \ \ 

    total_loss = 0

    \;

    \ \ \ \ 

    for data, label in data_iter():

    \;

    \ \ \ \ \ \ \ \ 

    with autograd.record():

    \;

    \ \ \ \ \ \ \ \ \ \ \ \ 

    output = net(data)

    \;

    \ \ \ \ \ \ \ \ \ \ \ \ 

    loss = square_loss(output, label)

    \;

    \ \ \ \ \ \ \ \ 

    loss.backward()

    \;

    \ \ \ \ \ \ \ \ 

    SGD(params, learning_rate)

    \;

    \;

    \ \ \ \ \ \ \ \ 

    total_loss += nd.sum(loss).asscalar()

    \;

    \ \ \ \ 

    print("Epoch %d, average loss: %f'' % (e, total_loss/num_examples))
  </code>

  \;

  <\code>
    Epoch 0, average loss: 7.946699

    Epoch 1, average loss: 0.099933

    Epoch 2, average loss: 0.001380

    Epoch 3, average loss: 0.000119

    Epoch 4, average loss: 0.000103
  </code>

  \<#8BAD\>\<#7EC3\>\<#5B8C\>\<#6210\>\<#540E\>\<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#6BD4\>\<#8F83\>\<#5B66\>\<#5230\>\<#7684\>\<#53C2\>\<#6570\>\<#548C\>\<#771F\>\<#5B9E\>\<#53C2\>\<#6570\>

  <code|In [11]:>

  <code|true_w, w>

  <code|Out[11]:>

  <\code>
    ([2, -3.4],

    \ [[ 1.99983037]

    \ \ [-3.39985394]]

    \ \<less\>NDArray 2x1 @cpu(0)\<gtr\>)
  </code>

  <code|In [12]:>

  <code|true_b, b>

  <code|Out[12]:>

  <\code>
    (4.2,

    \ [ 4.19959974]

    \ \<less\>NDArray 1 @cpu(0)\<gtr\>)
  </code>

  <section|\<#7ED3\>\<#8BBA\>>

  \<#6211\>\<#4EEC\>\<#73B0\>\<#5728\>\<#770B\>\<#5230\>\<#4EC5\>\<#4EC5\>\<#4F7F\>\<#7528\>NDArray\<#548C\>autograd\<#6211\>\<#4EEC\>\<#53EF\>\<#4EE5\>\<#5F88\>\<#5BB9\>\<#6613\>\<#5730\>\<#5B9E\>\<#73B0\>\<#4E00\>\<#4E2A\>\<#6A21\>\<#578B\>\<#3002\>

  <section|\<#7EC3\>\<#4E60\>>

  \<#5C1D\>\<#8BD5\>\<#7528\>\<#4E0D\>\<#540C\>\<#7684\>\<#5B66\>\<#4E60\>\<#7387\>\<#67E5\>\<#770B\>\<#8BEF\>\<#5DEE\>\<#4E0B\>\<#964D\>\<#901F\>\<#5EA6\>\<#FF08\>\<#6536\>\<#655B\>\<#7387\>\<#FF09\>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|优化|<tuple|7|?>>
    <associate|创建数据集|<tuple|2|?>>
    <associate|初始化模型参数|<tuple|4|?>>
    <associate|定义模型|<tuple|5|?>>
    <associate|损失函数|<tuple|6|?>>
    <associate|数据读取|<tuple|3|?>>
    <associate|训练|<tuple|8|?>>
    <associate|MathJax-Element-1-Frame|<tuple|1|?>>
    <associate|MathJax-Element-2-Frame|<tuple|1|?>>
    <associate|MathJax-Element-3-Frame|<tuple|1|?>>
    <associate|MathJax-Span-1|<tuple|1|?>>
    <associate|MathJax-Span-10|<tuple|1|?>>
    <associate|MathJax-Span-11|<tuple|1|?>>
    <associate|MathJax-Span-12|<tuple|1|?>>
    <associate|MathJax-Span-13|<tuple|1|?>>
    <associate|MathJax-Span-14|<tuple|1|?>>
    <associate|MathJax-Span-15|<tuple|1|?>>
    <associate|MathJax-Span-16|<tuple|1|?>>
    <associate|MathJax-Span-17|<tuple|1|?>>
    <associate|MathJax-Span-18|<tuple|1|?>>
    <associate|MathJax-Span-19|<tuple|1|?>>
    <associate|MathJax-Span-2|<tuple|1|?>>
    <associate|MathJax-Span-20|<tuple|1|?>>
    <associate|MathJax-Span-21|<tuple|1|?>>
    <associate|MathJax-Span-22|<tuple|1|?>>
    <associate|MathJax-Span-23|<tuple|1|?>>
    <associate|MathJax-Span-24|<tuple|1|?>>
    <associate|MathJax-Span-25|<tuple|1|?>>
    <associate|MathJax-Span-26|<tuple|1|?>>
    <associate|MathJax-Span-27|<tuple|1|?>>
    <associate|MathJax-Span-28|<tuple|1|?>>
    <associate|MathJax-Span-29|<tuple|1|?>>
    <associate|MathJax-Span-3|<tuple|1|?>>
    <associate|MathJax-Span-30|<tuple|1|?>>
    <associate|MathJax-Span-31|<tuple|1|?>>
    <associate|MathJax-Span-32|<tuple|1|?>>
    <associate|MathJax-Span-33|<tuple|1|?>>
    <associate|MathJax-Span-34|<tuple|1|?>>
    <associate|MathJax-Span-35|<tuple|1|?>>
    <associate|MathJax-Span-36|<tuple|1|?>>
    <associate|MathJax-Span-37|<tuple|1|?>>
    <associate|MathJax-Span-38|<tuple|1|?>>
    <associate|MathJax-Span-39|<tuple|1|?>>
    <associate|MathJax-Span-4|<tuple|1|?>>
    <associate|MathJax-Span-40|<tuple|1|?>>
    <associate|MathJax-Span-41|<tuple|1|?>>
    <associate|MathJax-Span-42|<tuple|1|?>>
    <associate|MathJax-Span-43|<tuple|1|?>>
    <associate|MathJax-Span-44|<tuple|1|?>>
    <associate|MathJax-Span-45|<tuple|1|?>>
    <associate|MathJax-Span-46|<tuple|1|?>>
    <associate|MathJax-Span-5|<tuple|1|?>>
    <associate|MathJax-Span-6|<tuple|1|?>>
    <associate|MathJax-Span-7|<tuple|1|?>>
    <associate|MathJax-Span-8|<tuple|1|?>>
    <associate|MathJax-Span-9|<tuple|1|?>>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-10|<tuple|9|?>>
    <associate|auto-11|<tuple|10|?>>
    <associate|auto-2|<tuple|1|?>>
    <associate|auto-3|<tuple|2|?>>
    <associate|auto-4|<tuple|3|?>>
    <associate|auto-5|<tuple|4|?>>
    <associate|auto-6|<tuple|5|?>>
    <associate|auto-7|<tuple|6|?>>
    <associate|auto-8|<tuple|7|?>>
    <associate|auto-9|<tuple|8|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|2fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|font-size|<quote|1.19>|1<space|2spc>\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>:\<#4ECE\>0\<#5F00\>\<#59CB\><locus|<id|%D568ADC8-D819A618>|<link|hyperlink|<id|%D568ADC8-D819A618>|<url|https://zh.gluon.ai/linear-regression-scratch.html#线性回归-----从0开始>>|>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|1fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>\<#7EBF\>\<#6027\>\<#56DE\>\<#5F52\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>\<#521B\>\<#5EFA\>\<#6570\>\<#636E\>\<#96C6\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>\<#6570\>\<#636E\>\<#8BFB\>\<#53D6\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>\<#521D\>\<#59CB\>\<#5316\>\<#6A21\>\<#578B\>\<#53C2\>\<#6570\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>\<#5B9A\>\<#4E49\>\<#6A21\>\<#578B\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>\<#635F\>\<#5931\>\<#51FD\>\<#6570\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|7<space|2spc>\<#4F18\>\<#5316\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|8<space|2spc>\<#8BAD\>\<#7EC3\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|9<space|2spc>\<#7ED3\>\<#8BBA\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|10<space|2spc>\<#7EC3\>\<#4E60\>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>