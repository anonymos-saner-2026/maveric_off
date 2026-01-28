#!/usr/bin/env python3
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from src.tools.real_toolkit import RealToolkit

def test_rotation():
    # Mock keys
    RealToolkit._SERPER_KEYS = ["KEY_EXPIRED", "KEY_WORKING"]
    RealToolkit._CURRENT_KEY_IDX = 0
    
    print(f"Initial key: {RealToolkit._get_current_serper_key()}")
    
    # Mock session
    with patch('requests.post') as mock_post:
        # First call return 403
        resp1 = MagicMock()
        resp1.status_code = 403
        
        # Second call return 200 with data
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"organic": [{"title": "Test Title", "link": "url", "snippet": "text"}]}
        
        mock_post.side_effect = [resp1, resp2]
        
        print("Executing search with expired first key...")
        results = RealToolkit._serper_search("test query")
        
        print(f"Results obtained: {len(results)} items")
        print(f"Current key index after rotation: {RealToolkit._CURRENT_KEY_IDX}")
        
        if RealToolkit._CURRENT_KEY_IDX == 1 and len(results) > 0:
            print("✅ TEST PASSED: Rotation triggered and search succeeded with next key.")
        else:
            print("❌ TEST FAILED: Rotation did not work as expected.")

if __name__ == "__main__":
    test_rotation()
