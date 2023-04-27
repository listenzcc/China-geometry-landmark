# 细粒度的中国地理数据分析（一）

如果有人让你解释什么是中国，你总不能告诉它这里是 960 万平方公里陆地面积的一只雄鸡，这太敷衍了，因为这么一大片土地上有着无数的细节。相信你也和我一样，想要了解这些细节。

本系列的数据来自 2018 年公开的细粒度的地理图像数据，它将中国的土地厘定成有意义的标签，包括住宅、商业、工业、交通和公共服务等。本系列将对其进行分析，尝试用定量的手段解释中国人文地理。

这是一个系列，随着写随着展开。也许我能发现什么有意思的东西。

---
- [细粒度的中国地理数据分析（一）](#细粒度的中国地理数据分析一)
  - [数据概况](#数据概况)
  - [细粒度初探](#细粒度初探)
  - [图表分析](#图表分析)
  - [附录：地块标记表](#附录地块标记表)


## 数据概况

本文的数据来自 2018 年公开的细粒度的地理图像数据，它将中国的土地厘定成有意义的标签，包括住宅、商业、工业、交通和公共服务等。

[Finer Resolution Observation and Monitoring - Global Land Cover](http://data.ess.tsinghua.edu.cn/)

首先要明确的是数据的规模很大，它将中国的陆地区域划分为 440,798 个地块，这些地块均有各类标记。将它绘制成点云的形式如下图所示，分别标示出了人口、工业、商业和交通。其中，人口、工业和商业分布高度吻合，交通建设稍显不足。这些种类地块的密度符合我国自东向西不断升高的地势。

由于点的大小是由地块的大小决定的，因此最后一张图中呈现出公园、绿地集中连片的特点，看上去观感不是很好，请不要介意。另外，还可以看到我国教育、医疗和绿地等公共设施主要集中在华北平原、长江下游、东北三省和广州等区域。

![Untitled](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/Untitled.png)

![Untitled](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/Untitled%201.png)

![Untitled](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/Untitled%202.png)

## 细粒度初探

接下来，由于原始数据极其密集，因此允许我们对不同的城市进行横向比较。比如左图是北京市的地块分布，右图是天津市的地块分布。看图说话，得到待证实的猜想如下：

- 北京的人口分布比较平均（作者吐槽其实是塞满了人），而天津的人口分布则集中于市中心区域；
- 天津的商业氛围差于北京，体现在 Bussiness office 数量明显更少；
- 令人意外的是，天津的工业并不差于北京，见 03 Industrial 部分。另外，还可以看到北京的工业地块是远离人口居住集中地块的，而天津的居住地块与工业地块高度吻合，这说明与北京相比，天津甚至还能算得上是一个工业为本的“传统”城市；
- 两市的交通枢纽建设都挺一言难尽的；
- 最后，北京的公共服务地块远超天津，甚至北京的公共服务地块与居住地块吻合度更高，天津的公共服务地块像是漫不经心地均匀洒在整个城市里一样，和居住地块没什么联系。这说明北京比天津更加像一个宜居的城市。

![110100.jpg](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/110100.jpg)

![120100.jpg](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/120100.jpg)

## 图表分析

接下来，我们不再拉踩天津，而是再将视角转回全国。我将数据库按照市级单位进行划分，之后统计每个市的不同地块面积所占的比例，再这些比例绘制成箱线图，如下图所示，上侧图的尺度为地块类，下侧图的尺度为地块细分。

从图中可见，我国城市中的工业地块占主导，居住地块其次，商业地块面积甚至小于公共服务地块。尽管如此，与居住地块相比，公共服务地块面积仍然较小。另外，我还发现交通地块中的机场是个很昂贵的东西，它虽然数量不大（与车站相比）但占地面积极其夸张，与公共服务的教育地块相近，也就是说机场的面积占比与大学相似。

![land-type-lvl1.png](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/land-type-lvl1.png)

![land-type-lvl2.png](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/land-type-lvl2.png)

## 附录：地块标记表

本文的数据来自 2018 年公开的细粒度的地理图像数据，它将中国的土地厘定成有意义的标签，包括住宅、商业、工业、交通和公共服务等。下图为地块标记表。

![Untitled](%E7%BB%86%E7%B2%92%E5%BA%A6%E7%9A%84%E4%B8%AD%E5%9B%BD%E5%9C%B0%E7%90%86%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%80%EF%BC%89%20f036e742830641139604faa7dc79ca6e/Untitled%203.png)