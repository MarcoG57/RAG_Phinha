"""Synthetic inventory and in-memory HTTP responses; these tests never use the network."""
import copy
import hashlib
import io
import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import download_dfl as downloader


def record(identifier, kind, payload=b'invented XML'):
    prefix = 'DFL_02_01_matchinformation' if kind == 'metadata' else 'DFL_04_03_positions_raw_observed'
    return {'id': identifier, 'name': f'{prefix}_DFL-COM-000001_DFL-MAT-J03WMX.xml',
            'download_url': f'https://ndownloader.figshare.com/files/{identifier}',
            'size': len(payload), 'computed_md5': hashlib.md5(payload).hexdigest(), 'is_link_only': False}


def inventory():
    return {'id': 28196177, 'version': 1, 'doi': '10.6084/m9.figshare.28196177.v1',
            'license': {'name': 'CC BY', 'url': 'https://creativecommons.org/licenses/by/4.0/'},
            'files': [record(51643475, 'metadata'), record(51643514, 'positions')]}


class InventoryTests(unittest.TestCase):
    def test_exact_pair_selected_from_public_inventory(self):
        pairs = downloader.inventory_pairs(inventory())
        self.assertEqual(set(pairs), {'DFL-MAT-J03WMX'})
        self.assertEqual(pairs['DFL-MAT-J03WMX']['metadata']['id'], 51643475)
        self.assertEqual(pairs['DFL-MAT-J03WMX']['positions']['id'], 51643514)

    def test_versioned_source_and_identity_are_required(self):
        self.assertEqual(downloader.API_URL, 'https://api.figshare.com/v2/articles/28196177/versions/1')
        for key, value in (('id', 28196178), ('version', 2), ('version', True),
                           ('doi', '10.6084/m9.figshare.28196177.v2')):
            altered = inventory()
            altered[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                downloader.inventory_pairs(altered)

    def test_inventory_license_paths_and_urls_must_match_contract(self):
        for mutate in (
            lambda item: item['license'].update(url='https://example.com/private-license'),
            lambda item: item['files'][0].update(name='../metadata.xml'),
            lambda item: item['files'][0].update(download_url='http://ndownloader.figshare.com/files/51643475'),
            lambda item: item['files'][0].update(download_url='https://example.com/files/51643475'),
            lambda item: item['files'][0].update(size=-1),
            lambda item: item['files'].pop(),
        ):
            value = copy.deepcopy(inventory())
            mutate(value)
            with self.assertRaises(ValueError):
                downloader.inventory_pairs(value)

    def test_md5_conflict_or_duplicate_pair_is_rejected(self):
        value = inventory()
        value['files'][0]['supplied_md5'] = '0' * 32
        with self.assertRaisesRegex(ValueError, 'MD5'):
            downloader.inventory_pairs(value)
        value = inventory()
        value['files'].append(value['files'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            downloader.inventory_pairs(value)


class Response(io.BytesIO):
    def __init__(self, value, headers=None):
        super().__init__(value)
        self.headers = headers or {}


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.target = Path(self.directory.name) / 'positions.xml'
        self.entry = downloader.inventory_pairs(inventory())['DFL-MAT-J03WMX']['positions']

    def test_stream_new_file_then_reuse_verified_without_network(self):
        with patch.object(downloader, '_open', return_value=Response(b'invented XML')) as request:
            self.assertEqual(downloader.download_file(self.entry, self.target), 'downloaded')
            self.assertEqual(self.target.read_bytes(), b'invented XML')
            self.assertFalse(self.target.with_suffix('.xml.part').exists())
            self.assertEqual(downloader.download_file(self.entry, self.target), 'reused')
            self.assertEqual(request.call_count, 1)

    def test_existing_wrong_content_same_size_is_preserved(self):
        self.target.write_bytes(b'wrong XML!!!')
        with patch.object(downloader, '_open') as request, self.assertRaisesRegex(ValueError, 'preserved'):
            downloader.download_file(self.entry, self.target)
        self.assertEqual(self.target.read_bytes(), b'wrong XML!!!')
        request.assert_not_called()

    def test_overlong_short_hash_or_header_mismatch_leaves_no_final(self):
        for payload, headers in ((b'too short', {}), (b'invented XML!', {}),
                                 (b'bad data!!!!', {}), (b'invented XML', {'Content-Length': '999'})):
            with self.subTest(payload=payload), patch.object(downloader, '_open', return_value=Response(payload, headers)):
                with self.assertRaises(ValueError):
                    downloader.download_file(self.entry, self.target)
                self.assertFalse(self.target.exists())
                self.assertFalse(self.target.with_suffix('.xml.part').exists())

    def test_existing_partial_is_preserved_without_network(self):
        part = self.target.with_suffix('.xml.part')
        part.write_bytes(b'previous interrupted run')
        with patch.object(downloader, '_open') as request, self.assertRaises(FileExistsError):
            downloader.download_file(self.entry, self.target)
        request.assert_not_called()
        self.assertEqual(part.read_bytes(), b'previous interrupted run')

    def test_atomic_publish_refuses_concurrent_destination(self):
        part = self.target.with_suffix('.xml.part')
        part.write_bytes(b'new')
        self.target.write_bytes(b'existing')
        with self.assertRaises(FileExistsError):
            downloader._publish(part, self.target)
        self.assertEqual(self.target.read_bytes(), b'existing')
        self.assertEqual(part.read_bytes(), b'new')

    def test_bounded_inventory_response(self):
        with patch.object(downloader, '_open', return_value=Response(b'x' * (downloader.MAX_INVENTORY_BYTES + 1))):
            with self.assertRaisesRegex(ValueError, 'bound'):
                downloader.fetch_inventory()

    def test_redirect_rejects_non_https_or_unapproved_storage(self):
        handler = downloader._SafeRedirect()
        request = downloader.urllib.request.Request('https://ndownloader.figshare.com/files/51643514')
        for url in ('http://s3-eu-west-1.amazonaws.com/file', 'https://example.com/file', 'https://user@s3.amazonaws.com/file'):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'redirect'):
                handler.redirect_request(request, None, 302, '', {}, url)

    def test_acquisition_writes_manifest_matching_loader_contract(self):
        root = Path(self.directory.name)
        responses = [Response(json.dumps(inventory()).encode()), Response(b'invented XML'), Response(b'invented XML')]
        with patch.object(downloader, 'ROOT', root), patch.object(downloader, '_open', side_effect=responses) as request:
            manifest = downloader.acquire(root / 'artifacts/dfl', None, accept_license='CC-BY4')
        parsed = tomllib.loads(manifest.read_text(encoding='utf-8'))
        row = parsed['matches'][0]
        self.assertEqual(row['id'], 'DFL-MAT-J03WMX')
        self.assertEqual(row['metadata'], 'J03WMX/metadata.xml')
        self.assertEqual(row['positions'], 'J03WMX/positions.xml')
        self.assertEqual(row['coordinate_origin'], 'center')
        self.assertEqual(row['fps'], 25)
        self.assertEqual(request.call_count, 3)
        receipt = json.loads((manifest.parent / 'download-manifest.json').read_text())
        self.assertTrue(receipt['acquisition_only'])
        self.assertEqual(receipt['license'], downloader.LICENSE_URL)

    def test_authority_and_outside_path_checked_before_network(self):
        with patch.object(downloader, '_open') as request:
            with self.assertRaisesRegex(ValueError, 'accept-license'):
                downloader.acquire(self.target.parent, None, accept_license=None)
            with self.assertRaisesRegex(ValueError, 'inside'):
                downloader.acquire(self.target.parent, None, accept_license='CC-BY4')
        request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
