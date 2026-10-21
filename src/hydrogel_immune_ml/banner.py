"""Build responsive web banners from the approved Illustrator export."""

import copy
import xml.etree.ElementTree as ET

from ..data import ROOT

NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', NS)
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')


def element(tag, attributes=None):
    return ET.Element(f'{{{NS}}}{tag}', attributes or {})


def build():
    source = ROOT / 'assets/scientific_3d/hero_source.svg'
    root = ET.parse(source).getroot()
    root.text = None
    root.set('width', '1500')
    root.set('height', '500')
    ET.ElementTree(root).write(ROOT / 'assets/banner.svg', encoding='utf-8', xml_declaration=True)

    mobile = element('svg', {'width': '520', 'height': '1540', 'viewBox': '0 0 520 1540'})
    for tag in ('title', 'desc'):
        mobile.append(copy.deepcopy(root.find(f'{{{NS}}}{tag}')))
    defs = copy.deepcopy(root.find(f'{{{NS}}}defs'))
    mobile.append(defs)
    artwork = copy.deepcopy(next(child for child in root if child.get('id') == 'source-layout'))
    artwork.set('id', 'mobile-artwork')
    for parent in artwork.iter():
        for child in list(parent):
            if child.tag == f'{{{NS}}}path' and child.get('d', '').startswith(('M501.351,236.937', 'M1043.694,236.937')):
                parent.remove(child)
    defs.append(artwork)
    for index, left in enumerate((-10, 504, 1015)):
        panel = element('svg', {'x': '0', 'y': str(index * 520), 'width': '520', 'height': '500',
                                'viewBox': f'{left} 0 520 500', 'overflow': 'hidden'})
        panel.append(element('use', {'href': '#mobile-artwork'}))
        mobile.append(panel)
        if index < 2:
            y = index * 520 + 500
            mobile.append(element('path', {'d': f'M260 {y + 1}v12m-4-4 4 4 4-4',
                                           'fill': 'none', 'stroke': '#a58baa', 'stroke-width': '1.7'}))
    ET.ElementTree(mobile).write(ROOT / 'assets/banner_mobile.svg', encoding='utf-8', xml_declaration=True)


if __name__ == '__main__':
    build()
