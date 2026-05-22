import tempfile
import unittest
from pathlib import Path

from prepare_cpt_text import clean_markdown, chunk_text


class TestCptPrepare(unittest.TestCase):
    def test_clean_markdown_removes_noise(self):
        sample = """
# Title

Intro text with a [link](http://example.com).

- item one
- [ ] item two

| A | B |
|---|---|
| 1 | 2 |

```python
print('code')
```

![alt](image.png)

Figure 1-1 Something

1

More text.
"""
        cleaned = clean_markdown(sample, keep_figures=False)
        self.assertIn("Title", cleaned)
        self.assertIn("Intro text with a link.", cleaned)
        self.assertIn("item one", cleaned)
        self.assertIn("item two", cleaned)
        self.assertIn("More text.", cleaned)
        self.assertNotIn("| A |", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertNotIn("![alt]", cleaned)
        self.assertNotIn("Figure 1-1", cleaned)

    def test_chunk_text(self):
        text = "one two three four five six seven eight nine ten"
        chunks = chunk_text(text, chunk_words=4, overlap_words=1)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunks))


if __name__ == "__main__":
    unittest.main()
