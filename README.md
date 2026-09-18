# GH200 3D stencil example (Miyabi-G, 4 nodes)

3D 7点ステンシル(拡散方程式のJacobi陽解法)を、Miyabi-Gの4ノード(GH200×4、1ノード1GPU)上で
MPI+CUDA、およびMPI+OpenACCの2通りで解く例題。両実装は分割方式・ハロー交換・実行モード・
出力フォーマットが同一になるよう作られており、`RESULT`行を突き合わせるだけで直接比較できる。

## 構成

- `src/stencil3d.cu` — CUDA版。プロセスグリッドは `MPI_Dims_create` + `MPI_Cart_create` で
  Px×Py×Pzに自動分割(明示指定も可)。分割数に上限はなく、64分割以上でも同じコードパスで動く
  (実機での大規模検証は未実施)。7点ステンシルは面隣接のみで完結するため、6面のみ交換すればよく、
  稜線・頂点の交換は不要。GPUカーネルはCUDA `__global__`関数として記述。
- `src/stencil3d_acc.cpp` — OpenACC版。ドメイン分割・ハロー交換・境界条件・2実行モードは
  CUDA版と完全に同一の設計。GPUカーネル部分だけを`#pragma acc parallel loop`に置き換え、
  CUDAストリーム(`s_interior`/`s_boundary`)はOpenACCの非同期キュー(`async(1)`/`async(2)`)に
  対応させている。MPI通信にはデバイスポインタをそのまま渡す(`#pragma acc host_data
  use_device`)ため、CUDA版同様GPU-aware MPIが前提。
- `Makefile` — CUDA版は `nvcc -ccbin mpicxx -arch=sm_90`、OpenACC版は
  `mpicxx -acc=gpu -gpu=cc90 -mp` でビルド(`make` で両方ビルドされる)。
- `job/payload.sh` — Miyabi-G向けPBSジョブスクリプト(`qsub`投入用)。同一ツールチェイン
  (`nvidia/26.3`)で両バイナリをビルドし、同一ドメイン・反復数で連続実行してCUDA版とOpenACC版を
  比較する。

## 実行モード

`./stencil3d NX NY NZ ITERS [PX PY PZ]` (OpenACC版は `./stencil3d_acc`)を1回実行すると、
**同一のMPIセッション内で**naive→overlapの順に両方走る(このクラスタのジョブラッパーは、
1プロセス1回のジョブ実行内で`MPI_Init`をやり直すことに対応していなかったため、2つのバイナリ
起動ではなく1プロセス内のモード切替えにしている):

- naive: pack → ブロッキング `MPI_Sendrecv`×6 → unpack → 全領域を1回の計算で更新。
- overlap: pack → 非ブロッキング `Isend/Irecv`×12 → ハロー未参照の内部領域を別ストリーム/
  非同期キューで計算しつつ通信を進行 → `Waitall` → unpack → 各面に接する厚さ1のシェルのみ再計算。

各モードの最後に `RESULT impl=<cuda|acc> ...` 行でGFLOP/s・実効帯域(GB/s)・通信時間を出力し、
CUDA版とOpenACC版を直接比較できる(`job/payload.sh`は両バイナリを1ジョブ内で連続実行するので、
出力ログを`grep RESULT`するだけで4行——CUDA naive/overlap、OpenACC naive/overlap——並ぶ)。

境界条件は全域Dirichlet(値0)で、ステンシルが触れない境界セルを単に更新しないことで実現している
(ノード間境界を跨がない場合も含め特別扱い不要)。`L2(u)`の出力は単調減少するはずで、崩れていれば
ハロー交換のバグを疑う簡易チェックになる。

## CUDA版とOpenACC版の違い

- **カーネル記述**: CUDA版は`__global__`関数+`<<<grid,block>>>`起動。OpenACC版は
  `#pragma acc parallel loop collapse(3)`によるディレクティブ指定で、コンパイラ(`nvc++`)が
  スレッド/ブロック分割を生成する。
- **非同期実行**: CUDA版のストリーム(`cudaStreamCreate`/`cudaStreamSynchronize`)に対応するのが
  OpenACC版の非同期キュー(`async(queue)`/`#pragma acc wait(queue)`)。
- **デバイスメモリ**: CUDA版は`cudaMalloc`で明示確保。OpenACC版はホスト側`malloc`した配列を
  `#pragma acc enter data create`でデバイスにも確保する構造化されない(unstructured)データ
  ライフタイムを使用。
- **初期化**: CUDA版はZ層ごとに`cudaMemcpy2D`でホスト→デバイス転送するが、OpenACC版はホスト側で
  ローカル領域全体をOpenMPで組み立ててから`#pragma acc update device`で一括転送する(定常状態の
  反復ループの性能比較には影響しない)。
- 上記以外(分割方式・ハロー交換・境界条件・2実行モード・GFLOPS/GB/s算出式)はすべて同一。

## Miyabi-Gでの実行(token消費)

トークン消費 = 経過時間(h) × ノード数 × 消費係数(Miyabi-G: 1.00)。

`debug-g`キュー(1〜16ノード、上限30分)を使い、`payload.sh`ではwalltime=10分に設定しているため、

```
4ノード × (10/60時間) × 1.00 ≒ 0.67トークン/回
```

100トークン枠なら140回以上テスト実行できる計算。ドメインサイズ(768³)・反復数(500)は
この時間内に余裕を持って収まるよう控えめに設定している(必要なら`payload.sh`内の`NX/NY/NZ/ITERS`を調整)。

### ジョブスクリプトの実行モデル

Miyabi-Gのqsubラッパーは、`#PBSWRAP SERIAL` / `#PBSWRAP PARALLEL` / `#PBSWRAP MODULE <modules...>`
の行でスクリプトをリージョン分割できる:

- `SERIAL`リージョン: rank 0だけで1回実行される。
- `PARALLEL`リージョン: 全ランクでそれぞれサンドボックス化(`bwrap`)されて実行される
  (`mpirun`を書く必要はない。書くとラッパーに拒否される — コメント中の文字列も含めて検出される)。
- 2リージョン間の同期はラッパーが保証する(SERIALが終わるまで他rankはPARALLELを開始しない)。
- `MODULE`: サンドボックス内では通常の`module load`シェルコマンドが効かない(`module: command not found`)ため、
  代わりに`#PBSWRAP MODULE nvidia/26.3 nv-hpcx`のように書くと、以降のリージョンにその環境が適用される。

各サンドボックスは`--tmpfs /work`・`--tmpfs /home`により、`/work`と`/home`が**ランクごとに独立した
空のtmpfs**に置き換わる(実体ではない)。ランク間で本当に共有される書き込み可能領域は`$HOME`(実体は
`/work/gz00/<group>/demo/runs/<jobid>`、同一ジョブの全ランクで同じホストディレクトリがbindされる)
だけなので、ビルド成果物や中間ファイルは**必ず`$HOME`配下**に置く。`job/payload.sh`では
最初の`SERIAL`でgit sync・両バイナリのビルドを行い、`PARALLEL`で`./stencil3d`を実行する。
1つの`PARALLEL`リージョンにつきmpirunで起動できるプログラムは1つだけ(同一リージョン内で
2回目の起動をすると`MPI_Init`が`getting local rank failed`で失敗する)なので、続けて
`SERIAL`→`PARALLEL`をもう一組はさんで`./stencil3d_acc`を実行している。

### 同期・投入フロー

Miyabi側で許可されている操作は `qstat` / `qsub` / `ls` / `tail` / `grep` / `qdel` のみで、
`git pull`のような操作は許可されていない。そのため**コード同期はジョブのpayload自身が行う**:

1. ここ(ローカル)で編集 → `github.com/cspp-lab/miyabi-gh200-stencil` にpush
2. `payload.sh`が実行時に同リポジトリを`git clone --depth 1`(初回)/`git pull --ff-only`(以降)
3. `ssh miyabi-agent 'qsub -N test -q debug-g -l select=4:mpiprocs=1:ompthreads=72 -l walltime=00:10:00 -W group_list=gz00 -j oe' < job/payload.sh`
4. `ssh miyabi-agent 'qstat <jobid>'` で確認、`ssh miyabi-agent 'tail <jobid> -n 50'` / `grep <jobid> -e Error -C 3` で結果確認

## 環境

- `module load nvidia/26.3 nv-hpcx`(NVIDIA HPC SDK + HPC-X、`mpicxx`/`nvcc`/OpenACC対応`nvc++`は
  このモジュールで揃う)
- リポジトリ: https://github.com/cspp-lab/miyabi-gh200-stencil (public)
