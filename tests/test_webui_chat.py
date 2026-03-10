from eaa.gui.chat import _parse_images_field


def test_parse_images_field_single_data_url():
    image = "data:image/png;base64,AAA"
    assert _parse_images_field(image) == [image]


def test_parse_images_field_json_list():
    image = '["data:image/png;base64,AAA","data:image/png;base64,BBB"]'
    assert _parse_images_field(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_python_literal_list():
    image = "['data:image/png;base64,AAA','data:image/png;base64,BBB']"
    assert _parse_images_field(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]


def test_parse_images_field_double_encoded_json_list():
    image = '"[\\"data:image/png;base64,AAA\\", \\"data:image/png;base64,BBB\\"]"'
    assert _parse_images_field(image) == [
        "data:image/png;base64,AAA",
        "data:image/png;base64,BBB",
    ]
