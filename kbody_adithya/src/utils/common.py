import logging
import typing

def assert_positive(
    logger:             logging.Logger,
    name:               str,
    value:              typing.Union[float, int],
):
    if value is not None and value <= 0.0:
        logger.error(f"Parameter {name} (value: {value}) should be positive.")

def ensure_choices(
    logger:         logging.Logger,
    name:    str,
    choice:         typing.Any,
    choices:        typing.Sequence[typing.Any],
) -> None:
    if choice not in choices:
        logger.error(f"The value given for {name} ({choice}) is invalid,"
            f" acceptable values are: {choices}"
        )
    return choice