package org.pytorch.demo.vision.view;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

public class DotView extends View {

   private static final float RADIUS = 4;
   private Paint mPaint;

   public DotView(Context context) {
      this(context, null);
   }

   public DotView(Context context, @Nullable AttributeSet attrs) {
      this(context, attrs, 0);
   }

   public DotView(Context context, @Nullable AttributeSet attrs, int defStyleAttr) {
      super(context, attrs, defStyleAttr);

      mPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
      mPaint.setColor(Color.GREEN);
   }


   @Override
   protected void onDraw(Canvas canvas) {
      canvas.drawCircle(RADIUS, RADIUS, RADIUS, mPaint);

   }
}
