﻿"Forecast price" 

بسمه تعالی
داده کاوی پیشبینی قیمت ده ارز و فلز گرانبها –
فهرست مطالب
۱ مقدمه و روشهای مورد استفاده -
۲ توضیح ابزار ها و محیط اجرا -
۳ تحلیل دادهها -
۴ پیش پردازش دادهها -
۵ پیشبینی قیمت با استفاده از روشهای آماری -
۶ ارزیابی مدل -
۷ مصور سازی نتایج -
۸ بهبود نتایج و روشهای دیگر -
۱ مقدمه و روشهای مورد استفاده -
برای پیشبینی قیمت در بازار های مختلف روشهای متعددی وجود دارد. استفاده از هر یک از این روشها بسته
به موقعیت های مختلف و با توجه به دادهها متفاوت میباشد، به عنوان مثال در بسیاری موارد به اینکه قیمت
افزایش پیدا میکند یا کاهش، بسنده میکنیم، که در این موارد میتوان از روشهای کلاس بندی دودویی ) binary classification ( استفاده نمود.
در موقعیت متفاوت دیگری نیاز داریم تا با توجه به دادههای از قبل موجود قیمت دقیق یک جنس در بازار را
پیشبینی و بررسی نماییم، که در این موارد روشهای رگرسیون میتوانند بسیار مفید باشند.
اما گونه دیگری از مسایل نیز وجود دارند که علاوه بر موارد گفته شده مسیله زمان نیز در آنها مطرح است. این
دسته از مسایل که به سری های زمانی معروف هستند به پیشبینی قیمت دقیق یک کالا در آینده و با توجه به
دادههای موجود میپردازد، ایده اصلی برای پیشبینی قیمت در چنین مسایلی استفاده از روشهای آمار و شبکه
عصبی است. به عنوان مثال میتوان روشهای Additive model و ARIMA )مخفف Auto-Regressive Integrated Moving Averages ( و LSTM )یک روش بر مبنای شبکهها عصبی ( را برای حل اینگونه
مسایل نام برد.
در این تحقیق من نیز از روش Additive model برای پیشبینی استفاده نمودم که در لینک زیر مبنای آن توضیح
داده شده است.
https://en.wikipedia.org/wiki/Additive_model
همچنین این روش نیاز به پیش پردازش خاص خود دارد و یا استفاده از ابزار ها و کتابخانهها مختص خود که
آنها را در هر بخش مرتبط آن توضیح میدهم.
تذکر: در هر مرحله تصاویری از کد پایتون مربوطه برای درک بهتر قرار داده شده است.
۲ توضیح ابزار ها و محیط اجرا -
در این تحقیق زبان برنامه نویسی پایتون مورد استفاده قرار گرفته است. همچنین کتابخانههای زیر نیز باید بر روی
سیستم عامل مربوطه نصب باشد. نام هر کدام و مورد کاربرد آن نیز بیان شده است.
برای دستکاری دادهها، خواندن از فایل :Pandas library csv
matplot library :مصور سازی دادهها و رسم نمودار
datatime library :تبدیل تاریخ از timestamp
numpy library :برای دستکاری دادهها و کار با آرایه های چند بعدی
fbprophet library : برای پیادهسازی روش آماری Additive model
scikit-learn library :برای ارزیابی نهایی مدل
تذکر: برای اجرای فایل ارسالی باید این کتابخانه ها نصب باشند و همچنین فایلهای داده و فایل اجرایی برنامه در
یک دایرکتوری قرار داشته باشند.
۳ تحلیل دادهها -
در این قسمت قصد تشکیل یک دیتاست برای پیشبینی قیمت در آینده داریم، بنابراین نیاز است که فایلهای
مختلف را برای کالا ها بررسی نماییم.
برای هر فلز یا ارز گرانبها سه فایل با نام های ticker و book و trades داده شده است. برای هر کدام در زیر
بیان نمودهام که چه اطلاعاتی برای پیشبینی مهم هستند و باید به دیتاست افزوده شود.)همراه با دلیل(
فایل ticker : از این فایل همه ستونها برای پیشبینی قیمت مهم میباشند به غیر از ستون ۸ و ۹ که مربوط به
حداکثر و حداقل قیمت روزانه میباشد که این دو ستون از دیتاست حذف شدهاند و میانگین آنها به عنوان قیمت
آن ارز یا فلز گرانبها مورد استفاده قرار گرفته است. این ستون همان ستونی است که در آینده باید پیشبینی شود.
در تصویر زیر کد پایتون مربوط به این کار را ملاحضه می نمایید.
فایل book : این فایل نشان دهنده سفارش ها است. یکی از ستونهای مهم در این فایل حجم سفارش است. به
دلیل اینکه این ستون حاوی مقادیر منفی نیز میباشد میتوان چنین استنباط نمود که به ازای مقادیر مثبت حجم
سفارش ما آن ارز یا فلز گرانبها را به بازار عرضه نموده ایم.)مانند عرضه دلار بانک مرکزی به بازار( و مقادیر منفی
حجم سفارش به معنی باز پس گرفتن آن ارز یا فلز گرانبها از بازار است.
حال موردی که مطرح است مجموع حجم سفارش ها برای یک زمان خاص است. که ممکن است این مجموع
حجم برای یک زمان خاص مثبت یا منفی به دست آید، برای هر کدام استنباط زیر مطرح است:
اگر حجم سفارش منفی بدست آید: یعنی ما آن ارز یا فلز گرانبها را بیش از حد به بازار عرضه نموده ایم و مشتری
برای آن وجود ندارد. )عرضه زیاد بوده است و باعث کاهش قیمت می شود.(
اگر حجم سفارش ها مثبت بدست آید: یعنی ما آن ارز یا فلز گرانبها را کمتر از تقاضا به بازار عرضه نموده ایم و
مشتری برای آن وجود دارد. )تقاضا زیاد بوده است و باعث افزایش قیمت می شود.(
بنابر اطلاعات بالا، باید برای هر زمان خاص حجم سفارشها محاسبه شود و به دیتاست اضافه گردد.
در تصویر زیر کد پایتون مربوط به این کار را ملاحضه می نمایید.
فایل trades : این فایل نیز حاوی معاملات انجام شده است و یکی از ستون هایی که میتواند در قیمت آینده تأثیر
گزار باشد، حجم معاملات انجام شده در یک زمان خاص است )به ازای هر سطر از این فایل(. بنابراین می توانیم
مانند فایل book به ازای هر زمان خاص )هر سطر از این فایل( مجموع حجم معاملات را محاسبه نماییم و به
دیتاست اضافه کنیم.
همچنین از این فایل مجموع قیمت معاملات انجام شده برای یک زمان خاص مهم است که آنرا نیز دقیقاً به طریق
بالا محاسبه می نماییم و به دیتاست اضافه میکنیم. کد آن در زیر آورده شده است.
بنابر توضیحات بالا دیتاست ما از ستونهای زیر تشکیل شده است:
ستون ۱: همان زمان ثبت اطلاعات با رزولوشون ۵ دقیقه است.
ستون ۲: همان ستون ۲ فایل ticker است.
ستون ۳: همان ستون ۳ فایل ticker است.
ستون ۴: همان ستون ۴ فایل ticker است.
ستون ۵: همان ستون ۵ فایل ticker است.
ستون ۶: همان ستون ۶ فایل ticker است.
ستون ۷: همان ستون ۷ فایل ticker است.
( ستون ۸ price (: میانگین ستونهای ۸ و ۹ فایل ticker است. )این ستون قیمت است و باید در آینده پیشبینی
شود.(
( ستون ۹ order_volume (: مجموع حجم سفارشهای داده شده برای آن زمان. )ستون اول دیتاست(
( ستون ۱۱ turnover (: مجموع حجم معاملات انجام شده برای آن زمان.
( ستون ۱۱ transaction  price (: مجموع قیمت معاملات انجام شده برای آن زمان.
در زیر چند سطر اول این دیتاست را در محیط terminal ملاحضه می نمایید.
۴ پیش پردازش دادهها -
در بخش پیش پردازش دادهها ابتدا نیاز است تا قالب زمان دادهها را از timestamp درآوریم و به current datetime تبدیل نماییم.
بحث دوم در مورد outlier ها در دادهها میباشد. مدل Additive از کتابخانه prophet تنها در صورتی که مقادیر
آنها را به np.pd.NaN )به صورت خیلی ساده منظور همان None در برنامه نویسی میباشد.( تبدیل کنیم قادر به
شناسایی آنها است و مانع از تغیر مقادیر پیشبینی در آینده میشود. البته در مستندات کتابخانه آمده بود که
بهترین راه برای پیش پردازش آنها حذف آنها است، منتها در این کار outlier ها حدف نشدهاند به دلیل اینکه در
بعضی موارد تعداد آنها بسیار کم بود و تأثیری نداشت اما برای بعضی از ارز ها یا فلز های گرانبها تعداد آنها زیاد
بود و باعث تغیر در نتیجه میشد که میتوان برای بهبود مدل آنها را حذف نمود.
در زیر کد مربوط به پیش پردازش دادهها را مشاهده می نمایید.
در نهایت پس از اعمال پیش پردازش های بالا، دادهها را به دو قسمت دادههای آموزشی و آزمایشی تقسیم کردم.
توجه نمایید که تنها ۱۱ درصد دادهها را برای آزمایش قرار دادم و مابقی آنها را برای آموزش مدل گزاشته ام.
۵ پیشبینی قیمت با استفاده از روشهای آماری -
در این قسمت به توضیح مدل آماری Additive model که برای پیشبینی استفاده شده است می پردازم.
در ابتدا موضوعی که مطرح است فهم این مدل است که چون فرصت مناسبی برای بیان آن در این گزارش نیست و
همچنین در استفاده از آن تأثیر چندانی ندارد، از بیان آن خودداری میکنم.
موضوع بعدی کتابخانه prophet و توابع آن است که برای استفاده از مدل آماری Additive model مورد استفاده
قرار می گیرد.
دو تابع مهم آن fit و predict است. در تابع اول به عنوان آرگومان ورودی یک دیتاست را دریافت می نماید و
برای آن یک مدل پیش گویی برای زمان های بعدی ایجاد می نماید. سپس با تابع predict و توسط آرگومان های
ورودی آن سری های زمانی آینده را پیشبینی میکند.
موضوع بعدی از این کتابخانه بازه زمان هایی است که پیشبینی میکند، این بازه میتواند سالانه، ماهانه، روزانه و
یا ساعتی باشد. اما قابلیت پیشبینی آینده در بازه دقیقهای را ندارد. با توجه به بررسی هایی که در مستندات این
کتابخانه داشتم در ورژن های آتی این قابلیت اضافه خواهد شد.
بنابراین در این پروژه تا ۷۲ ساعت بعد از آخرین زمانی که داده شده بود را پیشبینی نمودم و به جای اندازهگیری
۵۱ دقیقه بعدی با رزولوشن ۵ دقیقه، ۵۱ ساعت بعدی را با رزولوشن یک ساعت به یک ساعت اندازهگیری نمودم.
البته این مورد در نسخه های بعدی این کتابخانه برطرف خواهد شد. در زیر کد مربوط به این قسمت را مشاهده
می نمایید.
۶ ارزیابی مدل -
برای ارزیابی مدل از سنجه خطای میانگین مربعات استفاده شده است که همراه پیشبینی قیمت برای هر ازر یا فلز
گرانبها محاسبه میشود و در خروجی نمایش داده میشود.
۷ مصور سازی نتایج -
در قسمت مصور سازی نتایج برای هر ارز یا فلز گران بها یک نمودار رسم میشود. ستون عمودی مربوط به
قیمت است و خط افقی مربوط به زمان )در فرمت year-mounth-day (. در زیر تصویر مصور سازی نتایج را
برای ارز یا فلز گرانبهای A مشاهده می کنید. هر خط داخل این نمودار نشان دهنده مطلبی است که در زیر عکس
به آنها اشاره خواهم کرد.
نشان دهنده همان مقادیر واقعی قیمت در آن زمان است که درواقع به عنوان دادههای آموزشی ما آنها را با :Y
الگوریتم دادهایم. )رنگ مشکی داخل تصویر(
yhat : نشان دهنده همان مقادیر پیشبینی شده قیمت در آن زمان است. )رنگ آبی داخل تصویر(
true vlaue : این خط نشان دهنده مقادیر واقعی برای زمان های آینده است، این خط با توجه به دادههای تست
رسم شده است. )رنگ سبز(
predicted value : این خط نشان دهنده مقادیر پیشبینی شده برای زمان های آینده است. )رنگ قرمز(
همچنین در زیر نمودار مقادیر پیشبینی شده برای کالاهای A تا I محاسبه و رسم شده است.


۸ بهبود نتایج و روشهای دیگر -
در این تحقیق سعی شد تا با استفاده از مدل آماری Additive model یک پیشبینی نزدیک به واقعیت از آینده
داشته باشیم. در صورت نصب پیش نیاز ها و اجرای فایل پایتون ارسالی متوجه میشوید که این مدل برای
دادههای بدون outlier خوب عمل میکند و سنجه خطای میانگین مربعات برای آن مقادیر بین ۴ الی ۱۱ را محاسبه
میکند و در صورتی که دادهها دارای outlier باشند سنجه خطای میانگین مربعات حتی تا مقادیر بین ۳۱ الی ۴۱ هم
میرود.
بنابراین یکی از روشهای بهبود این مدل حذف outlier ها میباشد.
علاوه بر بحث بالا برای بهبود نتایج روشهای آماری و مبتنی بر شبکههای عصبی نیز میتوانند مفید واقع شوند.
ازین روشها میتوان به روش ARIMA )مخفف Auto-Regressive Integrated Moving Averages ( و LSTM
)یک روش بر مبنای شبکههای عصبی( نام برد.
پایان
تذکر ۱: برای اجرای فایل ارسالی باید این کتابخانه ها نصب باشند و همچنین فایلهای داده و فایل اجرایی برنامه
در یک دایرکتوری قرار داشته باشند.