import re
import unittest
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]


class DocumentLinks(unittest.TestCase):
    def test_local_markdown_links_resolve(self):
        files = [ROOT / 'README.md', ROOT / 'grader/README.md', *ROOT.glob('docs/**/*.md')]
        for source in files:
            for link in re.findall(r'\]\(([^)]+)\)', source.read_text()):
                if link.startswith(('https:', 'http:', '#')):
                    continue
                path = unquote(link.split('#')[0])
                if not path:
                    continue
                with self.subTest(source=str(source.relative_to(ROOT)), link=link):
                    self.assertTrue((source.parent / path).is_file(), f'失效的本地链接：{source}: {link}')


if __name__ == '__main__':
    unittest.main()
