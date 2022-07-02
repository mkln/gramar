## Graph Machine Regression


Minimal usage example:

```
devtools::install_github("mkln/gramar")
library(gramar)

#
# your data:
# train data: yin (vector), Xin (matrix, scaled)
# test data: Xout (matrix)

gramar_out <- gramar(y=yin, x=Xin, verbose=5, n_threads=16)
gramar_predict <- predict(gramar_out, newx=Xout, n_threads=16)
```
See full example [examples/boston.r](examples/boston.r)


Funding info: ERC 856506, NIH R01ES027498 and R01ES028804.

### [Poster at ISBA](poster/poster3648.pdf) World Meeting in Montreal, Jun 29, 2022

 - [.indd file for Gramar](poster/poster3648.indd)
 - ["template" files including fonts](poster/template)

![](poster/posterPNG.png?raw=true)
