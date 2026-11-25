# Sourced by the DP-527 archive jobs on host derpr. Not executable on its own.
#
# The archive is the r9700 HDD over NFS. Every job refuses to write unless it is
# really mounted: an unmounted /mnt/hdd-archive is an empty directory on derpr's
# root NVMe, the disk these copies exist to get data off.

ARCHIVE="${ARCHIVE:-/mnt/hdd-archive}"
MIN_FREE_MB="${MIN_FREE_MB:-5120}"

log() { printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$LOG" >&2; }

# The fstab entry is nofail, so a boot while r9700 is down leaves it unmounted.
# Try once before giving up rather than staying dark until someone notices.
require_archive() {
    mountpoint -q "$ARCHIVE" || mount "$ARCHIVE" 2>/dev/null
    if ! mountpoint -q "$ARCHIVE"; then
        echo "FATAL: $ARCHIVE is not mounted -- refusing to write to the root disk" >&2
        exit 1
    fi
    local free_mb
    free_mb=$(df -Pm "$ARCHIVE" | awk 'NR==2 {print $4}')
    if [ "${free_mb:-0}" -lt "$MIN_FREE_MB" ]; then
        echo "FATAL: only ${free_mb}MB free on $ARCHIVE, need ${MIN_FREE_MB}MB" >&2
        exit 1
    fi
}
