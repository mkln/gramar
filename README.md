## Graph Machine Regression


Minimal usage example:

```
devtools::install_github("mkln/gramar")
library(gramar)

#
# your data
#

gramar_out <- gramar(y=yin, x=Xin, verbose=5, n_threads=16
gramar_predict <- predict(gramar_out, newx=Xout, n_threads=16)
```
