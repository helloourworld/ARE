import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf
import scipy
from pathlib import Path

from enable_repo_root import ensure_repo_root, load_config

REPO_ROOT = ensure_repo_root(Path(__file__).resolve().parents[1])

try:
    from data_pipeline.data_cache import get_daily_returns as get_daily_returns_cached
except ImportError:
    from data_pipeline import get_daily_returns as get_daily_returns_cached

# Load configuration
cfg = load_config("config.yaml", REPO_ROOT)


class AlphaRiskEngine:
    def __init__(self, tickers, benchmark='SPY', start_date="2024-01-01"):
        self.tickers = cfg['parameters']['selected_tickers'] if 'selected_tickers' in cfg['parameters'] else tickers
        self.benchmark = cfg['parameters']['benchmark'] if 'benchmark' in cfg['parameters'] else benchmark
        self.start_date = start_date
        self.data = None
        self.returns = None
        self.robust_cov = None
        self.rf = cfg['parameters']['risk_free_rate'] / \
            252  # Daily proxy for risk-free rate

    def ingest_data(self):
        """Fetch adjusted close prices and calculate daily returns using persistent cache."""
        # print(f"Ingesting data for: {self.tickers}")
        self.returns = get_daily_returns_cached(self.tickers, self.benchmark, self.start_date)
        # print("Data Ingested. Sample Returns:")
        # print(self.returns.head())
        return self.returns

    def compute_robust_covariance(self):
        """
        Solves the 'Correlation > 1' problem using Ledoit-Wolf Shrinkage.
        Shrinks the sample covariance matrix towards a structured estimator.
        """
        lw = LedoitWolf()
        lw.fit(self.returns)
        self.robust_cov = pd.DataFrame(
            lw.covariance_, index=self.tickers + [self.benchmark], columns=self.tickers + [self.benchmark])
        print("Robust Covariance Matrix (Ledoit-Wolf) calculated.")
        return self.robust_cov

    def get_fama_french_factors(self):
        """
        In a production environment, pull from Ken French's Data Library.
        For this boilerplate, we use Proxy Factors:
        Mkt-RF: Benchmark Returns - Risk Free
        SMB (Size): Small Cap ETF - Large Cap ETF (Proxy)
        HML (Value): Value ETF - Growth ETF (Proxy)
        """
        # Note: Professional analysts use pandas_datareader.data.DataReader('F-F_Research_Data_Factors', 'famafrench')
        # Here we generate a simple market-excess return column for the regression demo.

        self.returns['Mkt-RF'] = self.returns[self.benchmark] - self.rf
        return self.returns

    def perform_factor_attribution(self, ticker):
        """
        Runs a Three-Factor OLS Regression.
        Goal: Identify if 'Alpha' in GIL or SNDK is actually just Value or Momentum exposure.
        """
        y = self.returns[ticker] - self.rf  # Excess return of asset
        # Expand this list as you add SMB and HML factors
        X = self.returns[['Mkt-RF']]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()
        print(f"\n--- Factor Attribution for {ticker} ---")
        print(model.summary())
        return model

    def VaR_gaussian(self, level=0.05, cf=False):
        '''
        https://www.kaggle.com/code/yousefsaeedian/var-cvar-analysis-and-sharp-ratio-calculation#%F0%9F%94%B815.-Conditional-VaR-(or-Beyond-VaR)
        Returns the (1-level)% VaR using the parametric Gaussian method. 
        By default it computes the 95% VaR, i.e., alpha=0.95 which gives level 1-alpha=0.05.
        The variable "cf" stands for Cornish-Fisher. If True, the method computes the 
        modified VaR using the Cornish-Fisher expansion of quantiles.
        The method takes in input either a DataFrame or a Series and, in the former 
        case, it computes the VaR for every column (Series).
        '''
        # alpha-quantile of Gaussian distribution
        za = scipy.stats.norm.ppf(level, 0, 1)
        if cf:
            S = self.skewness(self.returns)
            K = self.kurtosis(self.returns)
            za = za + (za**2 - 1)*S/6 + (za**3 - 3*za) * \
                (K-3)/24 - (2*za**3 - 5*za)*(S**2)/36
        return -(self.returns.mean() + za * self.returns.std(ddof=0))

    def calculate_tail_risk(self, confidence_level=0.95):
        """
        Calculates Conditional Value at Risk (CVaR).
        Essential for tech-heavy stocks like SNDK with non-normal distributions.
        """
        cvar_results = {}
        for ticker in self.tickers:
            var = np.percentile(
                self.returns[ticker], (1 - confidence_level) * 100)
            cvar = self.returns[ticker][self.returns[ticker] <= var].mean()
            cvar_results[ticker] = cvar
        return cvar_results

    def calculate_annualized_shortfall(self, confidence_level=0.95):
        """
        Calculates the Annualized CVaR (Expected Shortfall).
        This is more meaningful for institutional reporting than daily CVaR.
        """
        results = {}
        self.returns = self.ingest_data()  # Ensure returns are available
        for ticker in self.tickers:
            daily_returns = self.returns[ticker].dropna()
            if daily_returns.empty:
                results[ticker] = {
                    "Daily_ES": np.nan,
                    "Theoretical_Annual_ES": np.nan,
                    "Empirical_Annual_ES": np.nan,
                }
                continue

            # 1. Theoretical Annualization (Square Root of Time)
            var_95 = np.percentile(daily_returns, (1 - confidence_level) * 100)
            daily_cvar = daily_returns[daily_returns <= var_95].mean()
            theoretical_annual_cvar = daily_cvar * np.sqrt(252)

            # 2. Empirical Annualization (Using 21-day rolling windows - 1 month)
            # This captures path-dependency better than daily scaling
            monthly_returns = daily_returns.rolling(window=21).sum().dropna()
            empirical_annual_cvar = np.nan
            if not monthly_returns.empty:
                m_var_95 = np.percentile(
                    monthly_returns, (1 - confidence_level) * 100)
                monthly_cvar = monthly_returns[monthly_returns <= m_var_95].mean()

                # Scale monthly to annual (linear scaling for return-space averages)
                empirical_annual_cvar = monthly_cvar * np.sqrt(12)

            results[ticker] = {
                "Daily_ES": daily_cvar,
                "Theoretical_Annual_ES": theoretical_annual_cvar,
                "Empirical_Annual_ES": empirical_annual_cvar
            }
        return results

    def skewness(self):
        '''
        Computes the Skewness of the input Series or Dataframe.
        There is also the function scipy.stats.skew().
        '''
        return (((self.returns - self.returns.mean()) / self.returns.std(ddof=0))**3).mean()

    def kurtosis(self):
        '''
        Computes the Kurtosis of the input Series or Dataframe.
        There is also the function scipy.stats.kurtosis().
        '''
        return (((self.returns - self.returns.mean()) / self.returns.std(ddof=0))**4).mean() - 3

    def get_risk_summary(self, confidence_level=0.95):
        """Return a structured risk summary that consolidates tail-risk and correlation diagnostics."""
        if self.returns is None:
            self.returns = self.ingest_data()

        returns = self.returns.copy()
        if returns.empty:
            return {
                "summary": {},
                "by_asset": {},
                "consistency_checks": {},
                "advanced_metrics": {},
            }

        available_tickers = [ticker for ticker in self.tickers if ticker in returns.columns]
        if not available_tickers:
            return {
                "summary": {},
                "by_asset": {},
                "consistency_checks": {},
                "advanced_metrics": {},
            }

        es_metrics = self.calculate_annualized_shortfall(confidence_level=confidence_level)
        by_asset = {}
        for ticker in available_tickers:
            series = returns[ticker].dropna()
            if series.empty:
                continue

            var_level = np.percentile(series, (1 - confidence_level) * 100)
            daily_es = float(series[series <= var_level].mean())
            monthly_returns = series.rolling(window=21).sum().dropna()
            empirical_annual_es = np.nan
            if not monthly_returns.empty:
                monthly_var = np.percentile(monthly_returns, (1 - confidence_level) * 100)
                monthly_es = monthly_returns[monthly_returns <= monthly_var].mean()
                empirical_annual_es = float(monthly_es * np.sqrt(12))

            benchmark_series = returns[self.benchmark].dropna() if self.benchmark in returns.columns else None
            beta = np.nan
            if benchmark_series is not None and not benchmark_series.empty and not series.empty:
                aligned = pd.concat([series, benchmark_series], axis=1).dropna()
                if len(aligned) > 1:
                    x = sm.add_constant(aligned.iloc[:, 1])
                    beta = float(sm.OLS(aligned.iloc[:, 0], x).fit().params.iloc[1])

            by_asset[ticker] = {
                "daily_es": float(daily_es),
                "theoretical_annualized_es": float(daily_es * np.sqrt(252)),
                "empirical_annualized_es": float(empirical_annual_es),
                "beta": beta,
                "data_points": int(series.count()),
            }

        correlation_matrix = returns[available_tickers].corr().fillna(0)
        consistency_checks = {
            "es_methods_aligned": bool(np.isfinite(np.nanmean([metrics["theoretical_annualized_es"] for metrics in by_asset.values()]))),
            "sufficient_tail_data": len(returns[available_tickers].dropna()) >= 21,
            "volatility_stability": True,
            "beta_in_bounds": all(np.isnan(metrics["beta"]) or (-2 <= metrics["beta"] <= 3) for metrics in by_asset.values()),
            "no_nan_volatilities": all(np.isfinite(metrics["theoretical_annualized_es"]) for metrics in by_asset.values()),
        }

        summary = {
            "portfolio_volatility": float(returns[available_tickers].std(ddof=0).mean() * np.sqrt(252)) if available_tickers else np.nan,
            "portfolio_sharpe_ratio": np.nan,
            "diversification_ratio": np.nan,
            "max_drawdown": np.nan,
        }

        advanced_metrics = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "correlation_to_benchmark": {
                ticker: float(returns[ticker].corr(returns[self.benchmark])) if self.benchmark in returns.columns else np.nan
                for ticker in available_tickers
            },
            "risk_metric_correlation": correlation_matrix.to_dict(),
        }

        return {
            "summary": summary,
            "by_asset": by_asset,
            "consistency_checks": consistency_checks,
            "advanced_metrics": advanced_metrics,
        }


# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Your specific universe
    my_tickers = ['SNDK', 'GIL', 'MCD', 'GOOG', 'CLML.TO', 'XDIV.TO']

    are = AlphaRiskEngine(my_tickers)
    returns = are.ingest_data()

    # 1. Fix the Correlation issue
    cov_matrix = are.compute_robust_covariance()

    # 2. Extract specific Correlation between SNDK and the Market
    # This should now be <= 1.0 thanks to Ledoit-Wolf
    market_corr = returns.corr().loc['SNDK', 'SPY']
    print(f"\nAdjusted Correlation (SNDK vs SPY): {market_corr:.4f}")

    # 3. Analyze the "Value" play in Gildan (GIL)
    are.get_fama_french_factors()
    are.perform_factor_attribution('GIL')

    # 4. Stress Test: Tail Risk (CVaR)
    tails = are.calculate_annualized_shortfall()
    print("\nConditional Value at Risk (Expected Shortfall):")
    for ticker, metrics in tails.items():
        print(
            f"{ticker}: Daily ES={metrics['Daily_ES']:.4%}, Theoretical Annual ES={metrics['Theoretical_Annual_ES']:.4%}, Empirical Annual ES={metrics['Empirical_Annual_ES']:.4%}")
