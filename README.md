## Graph Machine Regression


Minimal usage example:

```
devtools::install_github("mkln/gramar")
library(gramar)

#
# your data:
# train data: yin (vector), Xin (matrix)
# test data: Xout (matrix)

gramar_out <- gramar(y=yin, x=Xin, verbose=5, n_threads=16
gramar_predict <- predict(gramar_out, newx=Xout, n_threads=16)
```


Funding info: ERC 856506, NIH R01ES027498 and R01ES028804.