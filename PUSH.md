# Push this folder to GitHub

Local path: `/home/ychua060/GEM-MM-github`  
Remote target: `https://github.com/SNOWTEAM2023/GEM-MM`

## Why push failed from this machine

The server SSH key is **not** authorized for `SNOWTEAM2023/GEM-MM`:

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICi/U92br2VGuoZhVHIxMiAvJH9l2QPB55LCCU6vpgT/ ychua060@lily-gpu07
```

## One-time setup

1. GitHub → org **SNOWTEAM2023** → repo **GEM-MM** → **Settings → Deploy keys**  
   (or add the key to your user SSH keys if you have write access)  
2. Paste the public key above, allow **write** access.  
3. From this machine:

```bash
cd /home/ychua060/GEM-MM-github
git remote add origin git@github.com:SNOWTEAM2023/GEM-MM.git   # if missing
git push -u origin main
```

If the remote already has commits, use:

```bash
git pull --rebase origin main
git push -u origin main
```

## Note on commit history

Commits in this folder are **real progressive setup commits** with honest
timestamps (init → docs → code → poster → polish). We do **not** backdate
history to fake weeks of work.
