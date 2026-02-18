import argparse
import getpass
import os
import queue
import tempfile
import threading
import zipfile

import paramiko


def upload_worker(sftp, upload_queue):
    """
    Worker thread that reads (local_path, remote_path) from queue,
    uploads via SFTP, and deletes the local temp file.
    """
    while True:
        task = upload_queue.get()
        if task is None:
            # Sentinel value received, stop processing
            upload_queue.task_done()
            break

        local_path, remote_path, batch_num = task
        try:
            print(f"    [>>>] Uploading Batch {batch_num} to remote...")
            sftp.put(local_path, remote_path)
            print(f"    [OK]  Batch {batch_num} upload complete.")
        except Exception as e:
            print(f"    [ERR] Failed to upload Batch {batch_num}: {e}")
        finally:
            # Clean up local temp file after upload (or failure)
            if os.path.exists(local_path):
                os.remove(local_path)
            upload_queue.task_done()


def create_zip_locally(files, source_root, batch_num):
    """Zips files to a temp file and returns the path."""
    print(f"[+] Zipping Batch {batch_num} ({len(files)} files)...")

    # Create temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    os.close(fd)  # Close file descriptor, we only need path

    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in files:
            # Preserve folder structure relative to the parent of source_root
            rel_path = os.path.relpath(file_path, os.path.dirname(source_root))
            zf.write(file_path, arcname=rel_path)

    return tmp_path


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Stream: Zip locally while uploading previous batch."
    )
    parser.add_argument("source_folder", help="Local folder to stream")
    parser.add_argument("remote_host", help="Remote hostname or IP")
    parser.add_argument("remote_user", help="Remote username")
    parser.add_argument("remote_path", help="Remote destination path (parent dir)")
    parser.add_argument(
        "chunk_mb", type=int, help="Target size of uncompressed data per zip in MB"
    )

    args = parser.parse_args()

    source_path = os.path.abspath(args.source_folder)
    if not os.path.exists(source_path):
        print(f"Error: Source {source_path} does not exist.")
        return

    password = getpass.getpass(f"Password for {args.remote_user}@{args.remote_host}: ")
    chunk_size = args.chunk_mb * 1024 * 1024

    # Connect SSH/SFTP
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(args.remote_host, username=args.remote_user, password=password)
        sftp = ssh.open_sftp()

        # Setup Remote Directory
        folder_name = os.path.basename(source_path.rstrip(os.sep))
        target_dir = f"{args.remote_path.rstrip('/')}/{folder_name}-zips"

        try:
            sftp.stat(target_dir)
        except FileNotFoundError:
            sftp.mkdir(target_dir)

        # Start Upload Worker Thread
        upload_q = queue.Queue()
        worker = threading.Thread(
            target=upload_worker, args=(sftp, upload_q), daemon=True
        )
        worker.start()

        # Main Loop: Walk and Zip
        batch_files = []
        current_batch_size = 0
        batch_count = 1

        for root, _, files in os.walk(source_path):
            for file in files:
                full_path = os.path.join(root, file)
                size = os.path.getsize(full_path)

                if current_batch_size + size > chunk_size and batch_files:
                    # 1. Create Zip (Blocking Main Thread)
                    zip_path = create_zip_locally(batch_files, source_path, batch_count)

                    # 2. Queue for Upload (Non-Blocking)
                    remote_zip_path = f"{target_dir}/batch_{batch_count:03d}.zip"
                    upload_q.put((zip_path, remote_zip_path, batch_count))

                    # 3. Reset for next batch immediately
                    batch_files = []
                    current_batch_size = 0
                    batch_count += 1

                batch_files.append(full_path)
                current_batch_size += size

        # Process final batch
        if batch_files:
            zip_path = create_zip_locally(batch_files, source_path, batch_count)
            remote_zip_path = f"{target_dir}/batch_{batch_count:03d}.zip"
            upload_q.put((zip_path, remote_zip_path, batch_count))

        # Wait for uploads to finish
        print("[*] Local processing done. Waiting for remaining uploads...")
        upload_q.put(None)  # Send sentinel to stop worker
        worker.join()  # Wait for worker to finish
        print("[+] Streaming complete.")

    except paramiko.AuthenticationException:
        print("Error: Authentication failed.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if "sftp" in locals():
            sftp.close()  # type: ignore
        if "ssh" in locals():
            ssh.close()


if __name__ == "__main__":
    main()
