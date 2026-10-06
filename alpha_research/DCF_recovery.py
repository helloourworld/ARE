"""Recovery-case DCF valuation for companies with temporarily negative FCF."""

import argparse


def calculate_recovery_dcf(
    fcf_forecast,
    growth_terminal,
    wacc,
    shares_out,
    debt,
    cash,
):
    """Return recovery-case price per share from an explicit FCF forecast."""
    if not fcf_forecast:
        raise ValueError("fcf_forecast must contain at least one annual value")
    if shares_out <= 0:
        raise ValueError("shares_out must be positive")
    if wacc <= growth_terminal:
        raise ValueError("wacc must be greater than growth_terminal")

    discount_periods = range(1, len(fcf_forecast) + 1)
    pv_fcf = sum(
        fcf / ((1 + wacc) ** period)
        for period, fcf in zip(discount_periods, fcf_forecast)
    )
    terminal_fcf = fcf_forecast[-1] * (1 + growth_terminal)
    terminal_value = terminal_fcf / (wacc - growth_terminal)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(fcf_forecast))
    equity_value = pv_fcf + pv_terminal_value + cash - debt
    return equity_value / shares_out


def parse_forecast(value):
    """Parse comma-separated annual FCF values expressed in millions."""
    try:
        forecast = [float(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated numbers, e.g. -2000,1000,4000") from error
    if not forecast:
        raise argparse.ArgumentTypeError("provide at least one FCF value")
    return forecast


def main():
    parser = argparse.ArgumentParser(description="Run an explicit recovery-case DCF.")
    parser.add_argument("--fcf-forecast", required=True, type=parse_forecast, help="annual FCF forecast in $M")
    parser.add_argument("--shares", required=True, type=float, help="shares outstanding in millions")
    parser.add_argument("--debt", required=True, type=float, help="total debt in $M")
    parser.add_argument("--cash", required=True, type=float, help="cash in $M")
    parser.add_argument("--wacc", type=float, default=0.08)
    parser.add_argument("--terminal-growth", type=float, default=0.035)
    args = parser.parse_args()
    value = calculate_recovery_dcf(
        args.fcf_forecast, args.terminal_growth, args.wacc,
        args.shares, args.debt, args.cash
    )
    print(f"Recovery-case DCF value: ${value:,.2f} per share")


if __name__ == "__main__":
    main()
