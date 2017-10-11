<TeXmacs|1.99.5>

<style|<tuple|generic|chinese|literate>>

<\body>
  <section|\<#5B89\>\<#88C5\>>

  First, configure your pip.

  <\verbatim-chunk|$HOME/.pip/pip.conf|false|false>
    [global]

    index-url = https://mirrors.ustc.edu.cn/pypi/web/simple

    format = columns
  </verbatim-chunk>

  Then:

  <\session|shell|default>
    <\output>
      Shell session inside TeXmacs pid = 28027
    </output>

    <\unfolded-io|Shell] >
      pip3 <code|install mxnet==0.11.0>
    <|unfolded-io>
      Collecting mxnet==0.11.0

      \ \ Using cached https://mirrors.ustc.edu.cn/pypi/web/packages/d7/1d/af82e37a65250c03ddfb89f5c096d673c167c9ee8122d219f0d7d4492a82/mxnet-0.11.0-py2.py3-none-manylinux1_x86_64.whl

      Collecting graphviz (from mxnet==0.11.0)

      \ \ Using cached https://mirrors.ustc.edu.cn/pypi/web/packages/9c/f8/a766d4f37c23483b9358fc7d21ee73a4c6ada7361ce88303110fcdfa3ee9/graphviz-0.8-py2.py3-none-any.whl

      Collecting numpy (from mxnet==0.11.0)

      \ \ Using cached https://mirrors.ustc.edu.cn/pypi/web/packages/0d/41/6c224571decd61c2578baedfdb0eec6283617c6679c35b20973f4e68aeaf/numpy-1.13.3-cp35-cp35m-manylinux1_x86_64.whl

      Installing collected packages: graphviz, numpy, mxnet

      Successfully installed graphviz-0.8 mxnet-0.11.0 numpy-1.13.3
    </unfolded-io>
  </session>

  <section|\<#68C0\>\<#67E5\>MXNet\<#662F\>\<#5426\>\<#53EF\>\<#7528\>>

  <\session|python|default>
    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      import mxnet as mx
    </input>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      <code|a = mx.nd.ones((2, 3))>
    </input>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      <code|b = a * 2 + 1>
    </input>

    <\unfolded-io>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|unfolded-io>
      b.asnumpy()
    <|unfolded-io>
      [[ 3. \ 3. \ 3.]

      \ [ 3. \ 3. \ 3.]]
    </unfolded-io>

    <\input>
      \<gtr\>\<gtr\>\<gtr\>\ 
    <|input>
      \;
    </input>
  </session>
</body>

<initial|<\collection>
</collection>>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-2|<tuple|2|?>>
    <associate|chunk-$HOME/.pip/pip.conf-1|<tuple|$HOME/.pip/pip.conf|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Installation>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Validate
      MXNet Installation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>