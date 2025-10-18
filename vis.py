import pandas as pd
import altair as alt


def plot_predictions(x, y, x_test, y_test_true, y_pred, y_pred_std=None, title=""):
    df_train = pd.DataFrame({"x": x.flatten(), "y": y.flatten()})
    df_test = pd.DataFrame(
        {
            "x": x_test.flatten(),
            "y_true": y_test_true.flatten(),
            "y_pred": y_pred.flatten(),
        }
    )
    if y_pred_std is not None:
        df_test["y_lower"] = (y_pred - y_pred_std).flatten()
        df_test["y_upper"] = (y_pred + y_pred_std).flatten()

    points = (
        alt.Chart(df_train)
        .encode(x="x", y="y")
        .mark_point(color="black", size=10, opacity=0.3)
    )
    line_true = (
        alt.Chart(df_test)
        .encode(x="x", y="y_true")
        .mark_line(color="blue", strokeDash=[5, 5], size=2)
        .properties(title=title)
    )
    line_pred = alt.Chart(df_test).mark_line(color="red", size=2).encode(x="x", y="y_pred")

    if y_pred_std is None:
        return points + line_true + line_pred

    band = (
        alt.Chart(df_test)
        .encode(x="x", y="y_lower", y2="y_upper")
        .mark_area(opacity=0.2, color="red")
    )
    return points + band + line_true + line_pred


def plot_losses(loss_df):
    return (
        alt.Chart(loss_df)
        .mark_line(point=True)
        .encode(x="epoch", y="loss")
        .properties(title="Training Loss over Epochs")
    )
