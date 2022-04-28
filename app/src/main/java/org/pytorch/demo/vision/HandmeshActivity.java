package org.pytorch.demo.vision;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewStub;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;


import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.annotation.WorkerThread;
import androidx.camera.core.ImageProxy;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Tensor;
import org.pytorch.demo.Constants;
import org.pytorch.demo.R;
import org.pytorch.demo.Utils;
import org.pytorch.demo.vision.utils.Dot;
import org.pytorch.demo.vision.view.DotView;
import org.pytorch.demo.vision.utils.YuvToRgbConverter;
import org.pytorch.demo.vision.view.ResultRowView;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Queue;
import java.lang.String;
import java.lang.Math;
import java.util.concurrent.ExecutorService;

public class HandmeshActivity extends AbstractCameraXActivity<HandmeshActivity.AnalysisResult> {

    public static final String INTENT_MODULE_ASSET_NAME = "INTENT_MODULE_ASSET_NAME";
    public static final String INTENT_INFO_VIEW_TYPE = "INTENT_INFO_VIEW_TYPE";
    public static final String SCORES_FORMAT = "%.2f";
    private static final int INPUT_TENSOR_WIDTH = 128;
    private static final int INPUT_TENSOR_HEIGHT = 128;
    private static final int TOP_K = 3;
    private static final int MOVING_AVG_PERIOD = 10;
    private static final String FORMAT_MS = "%dms";
    private static final String FORMAT_AVG_MS = "avg:%.0fms";
    private static final String FORMAT_FPS = "%.1fFPS";
    private ImageView mIvOrigin;
    private ImageView mIvSecondImage;
    private FrameLayout mDotContainer;
    private ExecutorService mExecutorService;
    private boolean mAnalyzeImageErrorState;
    private ResultRowView[] mResultRowViews = new ResultRowView[TOP_K];
    private TextView mFpsText;
    private TextView mMsText;
    private TextView mMsAvgText;
    private Module mModule;
    private String mModuleAssetName;
    private FloatBuffer mInputTensorBuffer;
    private Tensor mInputTensor;
    private long mMovingAvgSum = 0;
    private Queue<Long> mMovingAvgQueue = new LinkedList<>();

    public static Bitmap cropBitmap(Bitmap bitmap, int desigredWidth) {//从中间截取一个正方形
        int w = bitmap.getWidth(); // 得到图片的宽，高
        int h = bitmap.getHeight();
        int cropWidth = Math.min(w, h);// 裁切后所取的正方形区域边长
        cropWidth = Math.min(cropWidth, desigredWidth);
        return Bitmap.createBitmap(bitmap, (bitmap.getWidth() - cropWidth) / 2,
                (bitmap.getHeight() - cropWidth) / 2, cropWidth, cropWidth);
    }

    @Override
    protected int getContentViewLayoutId() {
        return R.layout.activity_image_classification;
    }

    @Override
    protected TextureView getCameraPreviewTextureView() {
        return ((ViewStub) findViewById(R.id.image_classification_texture_view_stub))
                .inflate()
                .findViewById(R.id.image_classification_texture_view);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        final ResultRowView headerResultRowView =
                findViewById(R.id.image_classification_result_header_row);
        headerResultRowView.nameTextView.setText(R.string.image_classification_results_header_row_name);
        headerResultRowView.scoreTextView.setText(R.string.image_classification_results_header_row_score);

        mIvOrigin = findViewById(R.id.iv_show_origin);

        mDotContainer = findViewById(R.id.flt_dot_container);
        mIvSecondImage = findViewById(R.id.iv_show_second);
        mResultRowViews[0] = findViewById(R.id.image_classification_top1_result_row);
        mResultRowViews[1] = findViewById(R.id.image_classification_top2_result_row);
        mResultRowViews[2] = findViewById(R.id.image_classification_top3_result_row);

        mFpsText = findViewById(R.id.image_classification_fps_text);
        mMsText = findViewById(R.id.image_classification_ms_text);
        mMsAvgText = findViewById(R.id.image_classification_ms_avg_text);


    }

    @Override
    protected void applyToUiAnalyzeImageResult(AnalysisResult result) {
        mMovingAvgSum += result.moduleForwardDuration;
        mMovingAvgQueue.add(result.moduleForwardDuration);
        if (mMovingAvgQueue.size() > MOVING_AVG_PERIOD) {
            mMovingAvgSum -= mMovingAvgQueue.remove();
        }

        for (int i = 0; i < TOP_K; i++) {
            final ResultRowView rowView = mResultRowViews[i];
            rowView.nameTextView.setText(result.topNClassNames[i]);
            rowView.scoreTextView.setText(String.format(Locale.US, SCORES_FORMAT,
                    result.topNScores[i]));
            rowView.setProgressState(false);
        }

        mMsText.setText(String.format(Locale.US, FORMAT_MS, result.moduleForwardDuration));
        if (mMsText.getVisibility() != View.VISIBLE) {
            mMsText.setVisibility(View.VISIBLE);
        }
        mFpsText.setText(String.format(Locale.US, FORMAT_FPS, (1000.f / result.analysisDuration)));
        if (mFpsText.getVisibility() != View.VISIBLE) {
            mFpsText.setVisibility(View.VISIBLE);
        }

        if (mMovingAvgQueue.size() == MOVING_AVG_PERIOD) {
            float avgMs = (float) mMovingAvgSum / MOVING_AVG_PERIOD;
            mMsAvgText.setText(String.format(Locale.US, FORMAT_AVG_MS, avgMs));
            if (mMsAvgText.getVisibility() != View.VISIBLE) {
                mMsAvgText.setVisibility(View.VISIBLE);
            }
        }
    }

    protected String getModuleAssetName() {
        if (!TextUtils.isEmpty(mModuleAssetName)) {
            return mModuleAssetName;
        }
        final String moduleAssetNameFromIntent =
                getIntent().getStringExtra(INTENT_MODULE_ASSET_NAME);
        mModuleAssetName = !TextUtils.isEmpty(moduleAssetNameFromIntent)
                ? moduleAssetNameFromIntent
                : "resnet18.pt";

        return mModuleAssetName;
    }

    @Override
    protected String getInfoViewAdditionalText() {
        return getModuleAssetName();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    @WorkerThread
    @Nullable
    protected AnalysisResult analyzeImage(ImageProxy image, int rotationDegrees) {
        if (mAnalyzeImageErrorState) {
            return null;
        }
        // 通过模型来识别数据
        try {
            if (mModule == null) {
                final String moduleFileAbsoluteFilePath = new File(Utils.assetFilePath(this,
                        getModuleAssetName())).getAbsolutePath();
                mModule = LiteModuleLoader.load(moduleFileAbsoluteFilePath);
                mInputTensorBuffer =
                        Tensor.allocateFloatBuffer(3 * INPUT_TENSOR_WIDTH * INPUT_TENSOR_HEIGHT);
                mInputTensor = Tensor.fromBlob(mInputTensorBuffer, new long[]{1, 3,
                        INPUT_TENSOR_HEIGHT, INPUT_TENSOR_WIDTH});
            }

            final long startTime = SystemClock.elapsedRealtime();

            Image frame = image.getImage(); // 一张图片

            float frame_width = image.getWidth();
            float frame_height = image.getHeight();
            // 进来的图是320*240，原因待确认，父类abstractCameraXActivty设置的resolution是224*224
            // 按最短的边，从左顶点开始，将图片成正方形
            Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(),
                    Bitmap.Config.ARGB_4444);
            YuvToRgbConverter converter = new YuvToRgbConverter(this);
            converter.yuvToRgb(frame, bitmap);


            runOnUiThread(() -> {
                mIvOrigin.setImageBitmap(bitmap);
                mIvSecondImage.setImageBitmap(bitmap);

            });


            Log.d("test_module",
                    "image , width:" + String.valueOf(frame_width) + ", heigh: " + frame_height +
                            ", bitmapImage width:" + bitmap.getWidth() + ", height:" + bitmap.getHeight());
            // 压缩成128 * 128, 模型接收的是128*128
//            Bitmap cropBitmap = cropBitmap(bitmap, 128);
//            TensorImageUtils.bitmapToFloatBuffer(cropBitmap, 0, 0, INPUT_TENSOR_WIDTH,
//                    INPUT_TENSOR_HEIGHT,
//                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
//                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
//                    mInputTensorBuffer, 0);

            TensorImageUtils.imageYUV420CenterCropToFloatBuffer(
                    frame, rotationDegrees,
                    INPUT_TENSOR_WIDTH, INPUT_TENSOR_HEIGHT,
                    TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                    TensorImageUtils.TORCHVISION_NORM_STD_RGB,
                    mInputTensorBuffer, 0);

            final long moduleForwardStartTime = SystemClock.elapsedRealtime();

            // 模型结果
            Log.d("test_module", "start model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ");

            IValue importTensor = IValue.from(mInputTensor);
            Log.d("test_module", mInputTensor.toString());
            final IValue outputTensor = mModule.forward(importTensor);

            // 转换成21 * 2的数组，没找到api直接转二维api，所以是tensor输出一维的array再转
            // py端pytorch输出的格式是：{"uvpred":[[[0,0],...*21]]}
            Map<String, IValue> mapValue = outputTensor.toDictStringKey();
            Iterator<Map.Entry<String, IValue>> iterator = mapValue.entrySet().iterator();

//            Log.e("test_module", mapValue.keySet().toString());
            IValue uvPred = mapValue.get("uv_pred");
            Tensor uvPredTensor = uvPred.toTensor();
//            Log.e("test_module", "print uvPred" + uvPredTensor.toString());
            long[] shape = uvPredTensor.shape();
//            Log.e("test_module", "shape: " + Arrays.toString(shape)); //[1, 21, 2]
            float[] valueList = uvPredTensor.getDataAsFloatArray();
//            Log.e("test_module",
//                    ", size: " + valueList.length); //42
            float[][] pointsArray = new float[(int) shape[1]][(int) shape[2]];
            for (int index_x = 0; index_x < shape[1]; index_x++) {
                for (int index_y = 0; index_y < shape[2]; index_y++) {
                    int index = (int) (index_y * shape[1] + index_x);
                    float value = (float) valueList[index];
                    pointsArray[index_x][index_y] = value;
                }
            }

            List<Dot> dots = new ArrayList<>();

            for (int index = 0; index < pointsArray.length; index++) {
                float x = pointsArray[index][0];
                float y = pointsArray[index][1];
                dots.add(new Dot(x, y));

            }
            runOnUiThread(() -> {
                mDotContainer.removeAllViews();
            });
            int width = mDotContainer.getMeasuredWidth();
            int measuredHeight = mDotContainer.getMeasuredHeight();
            for (Dot dot : dots) {
                int dotx = (int) (dot.x * width);
                int doty = (int) (dot.y * measuredHeight);
                Log.d("test_module:", "坐标点 x: " + dot.x + ", y: " + dot.y + " , dotx:" + dotx
                        + ",doty :" + doty + "， width：" + width + "， measuredHeight：" + measuredHeight);
                runOnUiThread(() -> {
                    DotView dotView = new DotView(HandmeshActivity.this);
                    dotView.setX(dotx);
                    dotView.setY(doty);
                    mDotContainer.addView(dotView);

                });

            }

            // 如何在图中显示这21个点，点的数值表示的是比例值，比如128*128的图，每个值*128就是像素点位置
            // ???
            // 弄一个小框显示标注21个点之后的图
            final long moduleForwardDuration =
                    SystemClock.elapsedRealtime() - moduleForwardStartTime;

            // 这是example的代码，原模型输出了n个类以及得分/概率，用于展示，可以丢弃›
            final String[] topKClassNames = new String[TOP_K];
            final float[] topKScores = new float[TOP_K];

            final long analysisDuration = SystemClock.elapsedRealtime() - startTime;
            return new AnalysisResult(topKClassNames, topKScores, moduleForwardDuration,
                    analysisDuration);
        } catch (Exception e) {
            Log.e(Constants.TAG, "Error during image analysis", e);
            mAnalyzeImageErrorState = true;
            runOnUiThread(() -> {
                if (!isFinishing()) {
                    showErrorDialog(v -> HandmeshActivity.this.finish());
                }
            });
            return null;
        }
    }

    @Override
    protected int getInfoViewCode() {
        return getIntent().getIntExtra(INTENT_INFO_VIEW_TYPE, -1);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mModule != null) {
            mModule.destroy();
        }
    }

    public Bitmap bytes2Bimap(byte[] b) {
        if (b.length != 0) {
            return BitmapFactory.decodeByteArray(b, 0, b.length);
        } else {
            return null;
        }
    }

    static class AnalysisResult {

        private final String[] topNClassNames;
        private final float[] topNScores;
        private final long analysisDuration;
        private final long moduleForwardDuration;

        public AnalysisResult(String[] topNClassNames, float[] topNScores,
                              long moduleForwardDuration, long analysisDuration) {
            this.topNClassNames = topNClassNames;
            this.topNScores = topNScores;
            this.moduleForwardDuration = moduleForwardDuration;
            this.analysisDuration = analysisDuration;
        }
    }
}
