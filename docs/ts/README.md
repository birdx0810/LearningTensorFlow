# Time Series

A _multivariate time series_ \(MTS\) could be defined as:

$$
X_{t,f} = \begin{pmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,f} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,f} \\
\vdots  & \vdots  & \ddots & \vdots  \\
x_{t,1} & x_{m,2} & \cdots & x_{t,f} 
\end{pmatrix}
$$

Where, $$t \in T$$is the sequence length, and $$f \in F$$ is the number of features. $$f=1$$is a special case of MTS, known as _univariate time series_. 

Time-Series Analysis could be divided into _frequency-domain_ and _time-domain_ methods.

* Frequency-Domain: 
  * Spectral Analysis
  * Wavelet Analysis
* Time-Domain: 
  * Auto-Correlation
  * Cross-Correlation

Time-Series data modeling could be categorized into _linear_ \(ARIMA-based\) and _non-linear_ \(ARCH-based\).

* Linear: The data could be modeled as a linear combination of past or future values or differences.
  * Auto-regressive \(AR\): The output is a linear regression of its $$M$$ previous values.
  * Moving Average \(MA\): The output is an $$N$$-point moving average of the input.
* Non-linear: The data is relatively chaotic and its variance would change over time.
  * Auto-regressive Conditional Heteroskedasticity \(ARCH\): describes the variance of the current error term or innovation as a function of the actual sizes of the previous time periods' error terms.



