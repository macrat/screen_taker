ディスプレイやスクリーンに投影した映像を撮影し、取得した映像から投影前の画像を推定したり、逆に手元にある画像を投影するとどうなるかを計算したりする。

# 仕組み
画像を投影したものを撮影する時に発生する劣化を1次関数による写像として捉えられないか、というのが基本的なアイデア。
写真の現像の時に使うカラーカーブのイメージ。

投影前の画像と投影後の画像のデータセットを作り、各ピクセルごとに最小自乗法で1次関数を作成している。
おそらく1次関数ではなくてロジスティック曲線にした方が性能が上がるのだけれど、計算が遅そうなので1次関数。

# 必要な環境
- python 3.5以降
- OpenCV 3
- Webカメラ

# 使い方
1. 画面のピクセル数に合わせてプログラムをいじる。

2. 起動する。

3. 認識させたい画面にscreenという名前のウィンドウを持って行き、エンターもしくはNキーを押す。

  認識範囲の自動認識が始まるので、ちょっと待つ。
  上手くいけば一瞬で終わる。上手くいかない場合は最初からやり直した方が良いかもしれない。

4. 認識範囲が確定したら、もう一度エンターもしくはNキーを押す。

  データセットの作成が始まるので、しばらく待つ。
  撮影が終了するとそのまま計算が始まるので、さらに待つ。

5. おわり。

  screenにはランダムな画像が表示される。
  expectというウィンドウには投影後の画面の予測映像が、diffというウィンドウには予測と実際の差分が表示される。

  diffに表示されている内容を使えば、画像を投影しなかった場合の投影面の状態を予測できる、かもしれない。

## gen\_graphの使い方
gen\_graph.pyを使うと、学習結果を可視化することが出来る。
main.pyの実行後にgen\_graph.pyを実行すると、以下の3枚の画像が生成される。

グラフの生成にはかなりの計算リソースが必要なので注意すること。

- coefficient.jpg

  `y = a * x + b`で言うところの、aの値。
  投影から撮影までのプロセスで発生する輝度や色の劣化が反映される。
  白く見えるところほど投影しようとした画像通りに投影され、黒く見えるところほど投影しようとした画像とは違うものが投影されたということになる。

- bias.jpg

  `y = a * x + b`で言うところの、bの値。
  投影した画像に関わりなく存在する値。画面の反射や、投影面自体の模様などが反映される（はず）。

- color\_lines.jpg

  1画素1画素のカラーカーブならぬカラー直線を重ねて描画したグラフ。
  全画素の`a * x + b`を計算して得られた直線を重ねたものとも言える。

  左下から右上に一本の直線が表示される状態が最も理想的なプロジェクタ+カメラの組み合わせということになる。
  横に一直線の線が現れたとしたら、何を投影しようとしてもカメラからの映像に変化がないということになる。
  直線の角度が急であればあるほど投影する画像の変化に敏感であり、角度が鈍ければ鈍いほど投影する画像が変化してもカメラから得られる画像には変化がない。