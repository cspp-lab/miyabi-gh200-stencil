# GH200 3D stencil example (Miyabi-G, 4 nodes)

3D 7点ステンシル(拡散方程式のJacobi陽解法)を、Miyabi-Gの4ノード(GH200×4、1ノード1GPU)上でMPI+CUDAで解く例題。

## 構成

- `src/stencil3d.cu` — 本体。プロセスグリッドは `MPI_Dims_create` + `MPI_Cart_create` で
  Px×Py×Pzに自動分割(明示指定も可)。分割数に上限はなく、64分割以上でも同じコードパスで動く
  (実機での大規模検証は未実施)。7点ステンシルは面隣接のみで完結するため、6面のみ交換すればよく、
  稜線・頂点の交換は不要。
- `Makefile` — `nvcc -ccbin mpicxx -arch=sm_90` でビルド。
- `job/payload.sh` — Miyabi-G向けPBSジョブスクリプト(`qsub`投入用)。

## 実行モード

コマンドライン引数 `MODE` で切替え、性能比較できる:

- `mode=0`(naive): pack → ブロッキング `MPI_Sendrecv`×6 → unpack → 全領域を1カーネルで更新。
- `mode=1`(overlap): pack → 非ブロッキング `Isend/Irecv`×12 → ハロー未参照の内部領域を別ストリームで
  計算しつつ通信を進行 → `Waitall` → unpack → 各面に接する厚さ1のシェルのみ再計算。

実行の最後に両モードとも `RESULT ...` 行でGFLOP/s・実効帯域(GB/s)・通信時間を出力する。
`job/payload.sh` は同一ジョブ内でnaive→overlapの順に実行し、直接比較できるようにしている。

境界条件は全域Dirichlet(値0)で、ステンシルが触れない境界セルを単に更新しないことで実現している
(ノード間境界を跨がない場合も含め特別扱い不要)。`L2(u)`の出力は単調減少するはずで、崩れていれば
ハロー交換のバグを疑う簡易チェックになる。

## Miyabi-Gでの実行(token消費)

トークン消費 = 経過時間(h) × ノード数 × 消費係数(Miyabi-G: 1.00)。

`debug-g`キュー(1〜16ノード、上限30分)を使い、`payload.sh`ではwalltime=10分に設定しているため、

```
4ノード × (10/60時間) × 1.00 ≒ 0.67トークン/回
```

100トークン枠なら140回以上テスト実行できる計算。ドメインサイズ(768³)・反復数(500)は
この時間内に余裕を持って収まるよう控えめに設定している(必要なら`payload.sh`内の`NX/NY/NZ/ITERS`を調整)。

### 同期・投入フロー

Miyabi側で許可されている操作は `qstat` / `qsub` / `ls` / `tail` / `grep` / `qdel` のみで、
`git pull`のような操作は許可されていない。そのため**コード同期はジョブのpayload自身が行う**:

1. ここ(ローカル)で編集 → `github.com/cspp-lab/miyabi-gh200-stencil` にpush
2. `payload.sh`が実行時に同リポジトリを`git clone --depth 1`(初回)/`git pull --ff-only`(以降)
3. `ssh miyabi-agent 'qsub -N test -q debug-g -l select=4:mpiprocs=1:ompthreads=72 -l walltime=00:10:00' < job/payload.sh`
4. `ssh miyabi-agent 'qstat -u z30105'` で確認、`ssh miyabi-agent 'tail <jobid> -n 50'` / `grep <jobid> -e Error -C 3` で結果確認

## 要確認・要調整

- `module load nvidia nvmpi` は仮のプレースホルダ。Miyabi-Gの実際のモジュール名は
  `module avail` の結果に合わせて `job/payload.sh` を書き換える必要がある。
- `github.com/cspp-lab/miyabi-gh200-stencil` はこのセッションでは未作成。push前にリポジトリ作成が必要。
