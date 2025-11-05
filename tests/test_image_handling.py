# tests/test_image_handling.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest
from src.interfaces.discord_bot import get_image_url


@pytest.mark.asyncio
async def test_get_image_url_with_attachment():
    """
    Tests that get_image_url correctly extracts an image URL from a message attachment.
    """
    # Create a mock attachment
    mock_attachment = MagicMock()
    mock_attachment.content_type = 'image/png'
    mock_attachment.url = 'http://example.com/image.png'

    # Create a mock message with the attachment
    mock_message = MagicMock()
    mock_message.attachments = [mock_attachment]
    mock_message.content = 'This is a message with an attachment.'

    # Call the function and assert the result
    image_url = await get_image_url(mock_message)
    assert image_url == 'http://example.com/image.png'


@pytest.mark.asyncio
async def test_get_image_url_with_url_in_content():
    """
    Tests that get_image_url correctly extracts an image URL from the message content.
    """
    # Create a mock message with a URL in the content
    mock_message = MagicMock()
    mock_message.attachments = []
    mock_message.content = 'Check out this image: http://example.com/image.jpg'

    # Call the function and assert the result
    image_url = await get_image_url(mock_message)
    assert image_url == 'http://example.com/image.jpg'


@pytest.mark.asyncio
async def test_get_image_url_with_no_image():
    """
    Tests that get_image_url returns None when there is no image in the message.
    """
    # Create a mock message with no image
    mock_message = MagicMock()
    mock_message.attachments = []
    mock_message.content = 'This is a message with no image.'

    # Call the function and assert the result
    image_url = await get_image_url(mock_message)
    assert image_url is None


@pytest.mark.asyncio
async def test_get_image_url_prefers_attachment_over_url():
    """
    Tests that get_image_url prefers the attachment URL over a URL in the content.
    """
    # Create a mock attachment
    mock_attachment = MagicMock()
    mock_attachment.content_type = 'image/gif'
    mock_attachment.url = 'http://example.com/attachment.gif'

    # Create a mock message with both an attachment and a URL in the content
    mock_message = MagicMock()
    mock_message.attachments = [mock_attachment]
    mock_message.content = 'Here is an attachment and a URL: http://example.com/content.webp'

    # Call the function and assert the result
    image_url = await get_image_url(mock_message)
    assert image_url == 'http://example.com/attachment.gif'

@pytest.mark.asyncio
async def test_get_image_url_handles_multiple_attachments():
    """
    Tests that get_image_url returns the first valid image attachment URL.
    """
    # Create mock attachments
    mock_attachment_1 = MagicMock()
    mock_attachment_1.content_type = 'image/jpeg'
    mock_attachment_1.url = 'http://example.com/image1.jpeg'

    mock_attachment_2 = MagicMock()
    mock_attachment_2.content_type = 'image/bmp'
    mock_attachment_2.url = 'http://example.com/image2.bmp'

    # Create a mock message with multiple attachments
    mock_message = MagicMock()
    mock_message.attachments = [mock_attachment_1, mock_attachment_2]
    mock_message.content = ''

    # Call the function and assert the result
    image_url = await get_image_url(mock_message)
    assert image_url == 'http://example.com/image1.jpeg'


@pytest.mark.asyncio
async def test_get_image_url_ignores_non_image_attachments():
    """
    Tests that get_image_url ignores attachments that are not images.
    """
    # Create mock attachments
    mock_attachment_1 = MagicMock()
    mock_attachment_1.content_type = 'text/plain'
    mock_attachment_1.url = 'http://example.com/text.txt'

    mock_attachment_2 = MagicMock()
    mock_attachment_2.content_type = 'image/png'
    mock_attachment_2.url = 'http://example.com/image.png'

    # Create a mock message with a non-image attachment and an image attachment
    mock_message = MagicMock()
    mock_message.attachments = [mock_attachment_1, mock_attachment_2]
    mock_message.content = ''

    # Call the function and assert the result
    image_url = await get_image_url(mock_message)
    assert image_url == 'http://example.com/image.png'
