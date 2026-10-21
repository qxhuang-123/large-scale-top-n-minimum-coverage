"""Check page geometry, extractable text and embedded fonts in exported PDFs."""
from pathlib import Path
import json

import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent


def embedded(font):
    obj = font.get_object()
    descendants = obj.get('/DescendantFonts')
    if descendants:
        obj = descendants[0].get_object()
    descriptor = obj.get('/FontDescriptor')
    if descriptor is None:
        return False
    descriptor = descriptor.get_object()
    return any(descriptor.get(key) is not None
               for key in ['/FontFile', '/FontFile2', '/FontFile3'])


def main():
    paths = sorted((ROOT / 'figures').glob('*.pdf'))
    paths += sorted((ROOT / 'tables').glob('*.pdf'))
    paths += [ROOT / 'Lean_Main_Preview.pdf']
    checks = []
    for path in paths:
        reader = PdfReader(str(path))
        fonts = []
        for page in reader.pages:
            resources = page.get('/Resources', {}).get_object()
            for font in resources.get('/Font', {}).values():
                fonts.append(embedded(font))
        assert fonts and all(fonts), f'Unembedded font in {path.name}'
        with pdfplumber.open(path) as doc:
            assert len(doc.pages) == len(reader.pages)
            words = 0
            for page in doc.pages:
                page_words = page.extract_words()
                words += len(page_words)
                for word in page_words:
                    assert word['x0'] >= -0.5 and word['x1'] <= page.width + 0.5
                    assert word['top'] >= -0.5 and word['bottom'] <= page.height + 0.5
            assert words > 0, f'No extractable text in {path.name}'
        checks.append({'path': str(path.relative_to(ROOT)), 'pages': len(reader.pages),
                       'font_entries': len(fonts), 'all_fonts_embedded': True,
                       'extractable_words': words, 'text_inside_page': True})
    result = {'passed': True, 'pdfs': len(checks), 'checks': checks}
    (ROOT / 'qa' / 'pdf_checks.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(f"PDF checks passed: {len(checks)} files.")


if __name__ == '__main__':
    main()
