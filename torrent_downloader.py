import libtorrent as lt
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import sqlite3
import time

class TorrentDownloader:
    def __init__(self):
        self.session = lt.session()

    def download_torrent(self, torrent_file, save_path, files_to_download):
        info = lt.torrent_info(torrent_file)
        h = self.session.add_torrent({'ti': info, 'save_path': save_path})

        file_priorities = [0] * info.num_files()
        for i, file in enumerate(info.files()):
            if file.path in files_to_download:
                file_priorities[i] = 1

        h.prioritize_files(file_priorities)
        while not h.is_seed():
            # This loop will block until download is complete. You can add more detailed progress or status checks here if needed.
            time.sleep(1)

class MD5Database:
    def __init__(self, db_path):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()

    def get_all_md5(self):
        self.cursor.execute("SELECT md5 FROM compact")
        return [item[0] for item in self.cursor.fetchall()]

class TorrentManager:
    def __init__(self, torrent_list, save_path, max_concurrent_downloads, db_path):
        self.queue = Queue()
        for torrent in torrent_list:
            self.queue.put(torrent)
        self.downloader = TorrentDownloader()
        self.save_path = save_path
        self.max_concurrent_downloads = max_concurrent_downloads
        self.db = MD5Database(db_path)

    def process_queue(self):
        files_to_download = self.db.get_all_md5()
        with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
            while not self.queue.empty():
                torrent = self.queue.get()
                executor.submit(self.downloader.download_torrent, torrent, self.save_path, files_to_download)

if __name__ == "__main__":
    torrents = ["path/to/torrent1.torrent", "path/to/torrent2.torrent", ...]
    MAX_CONCURRENT_DOWNLOADS = 10
    DB_PATH = "path_to_your_sqlite_database.db"
    SAVE_PATH = "path_to_save_downloaded_files"
    manager = TorrentManager(torrents, SAVE_PATH, MAX_CONCURRENT_DOWNLOADS, DB_PATH)
    manager.process_queue()
