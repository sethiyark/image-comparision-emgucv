using System;
using System.IO;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.XFeatures2D;

namespace OpenCv_CSharp1
{
    public partial class Form1 : Form
    {
        string master = "";
        string test = "";
        Bitmap img1 = null;
        Bitmap img2 = null;

        public Form1()
        {
            InitializeComponent();
        }

        private void linkLabel1_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            openFileDialog1.FileName = "";
            openFileDialog1.Title = "Images";
            openFileDialog1.Filter = "All Images|*.jpg; *.bmp; *.png";
            openFileDialog1.ShowDialog();
            if (openFileDialog1.FileName.ToString() != "")
            {
                master = openFileDialog1.FileName.ToString();
                pictureBox1.Image = null;
                img1 = new Bitmap(master);
                pictureBox1.Visible = true;
                pictureBox1.Image = img1;
                //pictureBox1.Height = img1.Height;
                //pictureBox1.Width = img1.Width;
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            }


        }

        private void linkLabel2_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            openFileDialog2.FileName = "";
            openFileDialog2.Title = "Images";
            openFileDialog2.Filter = "All Images|*.jpg; *.bmp; *.png";
            openFileDialog2.ShowDialog();
            if (openFileDialog2.FileName.ToString() != "")
            {
                test = openFileDialog2.FileName.ToString();
                pictureBox2.Image = null;

                img2 = new Bitmap(test);
                pictureBox2.Visible = true;
                pictureBox2.Image = img2;
                //pictureBox2.Height = img2.Height;
                //pictureBox2.Width = img2.Width;
                pictureBox2.SizeMode = PictureBoxSizeMode.StretchImage;
            }

        }

        class myClass
        {
            string master;
            string test;
            string bg;
            public myClass(string master, string test, string bg)
            {
                this.master = master;
                this.test = test;
                this.bg = bg;
            }
            public void compare()
            {

                using (Mat des_t = new Mat())
                using (Image<Gray, byte> theMaster = new Image<Gray, byte>(master))
                using (Image<Gray, byte> theTest = new Image<Gray, byte>(test))
                using (Image<Gray, byte> mask = new Image<Gray, byte>(theMaster.Size))
                using (Mat des_m = new Mat())
                using (VectorOfKeyPoint kp_m = new VectorOfKeyPoint())
                using (VectorOfKeyPoint kp_t = new VectorOfKeyPoint())
                {
                    //SURF br = new SURF(12500);
                    Brisk br = new Brisk(75, 8, 14.28f);
                    //Image<Gray, byte> backGround = new Image<Gray, byte>(theMaster.Size);

                    //backGround = new Image<Gray, byte>(bg);

                    System.Diagnostics.Stopwatch timer = System.Diagnostics.Stopwatch.StartNew();
                    //CvInvoke.Subtract(backGround, theMaster, mask);
                    //Image<Gray, byte> theMask = mask.Convert<Gray, byte>();
                    //Image<Gray, byte> thetMask = tmask.Convert<Gray, byte>();
                    //theMask = theMask.ThresholdBinary(new Gray(120), new Gray(255));
                    //CvInvoke.Imshow("Mask", mask.Resize(0.2, Emgu.CV.CvEnum.Inter.Nearest));
                    
                    try
                    {
                        br.DetectAndCompute(theMaster, null, kp_m, des_m, false);
                        br.DetectAndCompute(theTest, null, kp_t, des_t, false);
                        br = null;
                    }
                    catch (CvException ex)
                    {
                        MessageBox.Show(ex.ToString());
                        throw ex;
                    }
                    finally
                    {
                        if (br != null)
                            br.Dispose();
                    }


                    BFMatcher bf = new BFMatcher(DistanceType.L2);
                    VectorOfVectorOfDMatch raw_matches = new Emgu.CV.Util.VectorOfVectorOfDMatch();

                    bf.Add(des_t);

                    try
                    {
                        bf.KnnMatch(des_m, raw_matches, 2, null);
                        bf = null;
                    }
                    catch (Exception e1)
                    {
                        MessageBox.Show(e1.Message);
                        throw;
                    }

                    MDMatch[][] dmatches_tm = raw_matches.ToArrayOfArray();

                    List<MDMatch> good_matches = new List<MDMatch>();
                    foreach (MDMatch[] m in dmatches_tm)
                    {
                        if (m[0].Distance < 0.9 * m[1].Distance && m[0].Distance > 0.7 * m[1].Distance)
                            good_matches.Add(m[0]);
                    }

                    List<int> qidx_t = new List<int>();
                    List<int> tidx_t = new List<int>();
                    int final_dist_t = 0;
                    int oct_count_t = 0;

                    foreach (MDMatch m in good_matches)
                    {
                        var a = kp_m[m.QueryIdx];
                        var b = kp_t[m.TrainIdx];
                        if ((Convert.ToInt32(a.Octave) == Convert.ToInt32(b.Octave))
                            if(Math.Abs(a.Size - b.Size)<1))
                        {
                            oct_count_t += 1;
                            tidx_t.Add(m.TrainIdx);
                            qidx_t.Add(m.QueryIdx);
                        }
                    }
                    
                    for (int i = 0; i < qidx_t.Count; i++)
                    {
                        int q_base = qidx_t[i];
                        int t_base = tidx_t[i];
                        final_dist_t--;

                        for (int j = i; j < qidx_t.Count; j++)
                        {
                            float q = eucledianDist(kp_m[qidx_t[j]].Point, kp_m[q_base].Point);
                            float t = eucledianDist(kp_t[tidx_t[j]].Point, kp_t[t_base].Point);

                            if (Math.Abs(q - t) <= ( 0.06 * q ))
                                final_dist_t++;
                        }
                    }

                    
                    //Image<Bgr, byte> resultImg = new Image<Bgr, byte>(theMaster.Size);
                    //Features2DToolbox.DrawMatches(theMaster, kp_m, theTest, kp_t, raw_matches, resultImg, new MCvScalar(0), new MCvScalar(155));
                    //CvInvoke.Imshow("Res", resultImg.Resize(0.25, Emgu.CV.CvEnum.Inter.Nearest));
                    
                    float raw_len = dmatches_tm.Length;
                    float final_dist = final_dist_t;
                    float per = (final_dist / raw_len) * 100;
                    
                    long time_taken = timer.ElapsedMilliseconds;
                    MessageBox.Show("Result " + per + "\nTime: " + time_taken + 
                        "\nmKP" + kp_m.Size + "\ntKP" + kp_t.Size + 
                        "\ngood_kp" + good_matches.Count + "\noct_cnt" + oct_count_t + "\ndist "+ final_dist);
                }
            }

            public string compareAll()
            {

                using (Mat des_t = new Mat())
                using (Image<Gray, byte> theMaster = new Image<Gray, byte>(master))
                using (Image<Gray, byte> theTest = new Image<Gray, byte>(test))
                using (Image<Gray, byte> mask = new Image<Gray, byte>(theMaster.Size))
                using (Mat des_m = new Mat())
                using (VectorOfKeyPoint kp_m = new VectorOfKeyPoint())
                using (VectorOfKeyPoint kp_t = new VectorOfKeyPoint())
                {
                    //SURF br = new SURF(10000, 8, 4);
                    Brisk br = new Brisk(80, 7, 4f);
                    //Image<Gray, byte> backGround = new Image<Gray, byte>(theMaster.Size);

                    //backGround = new Image<Gray, byte>(bg);

                    System.Diagnostics.Stopwatch timer = System.Diagnostics.Stopwatch.StartNew();
                    //CvInvoke.Subtract(backGround, theMaster, mask);
                    //Image<Gray, byte> theMask = mask.Convert<Gray, byte>();
                    //Image<Gray, byte> thetMask = tmask.Convert<Gray, byte>();
                    //theMask = theMask.ThresholdBinary(new Gray(120), new Gray(255));
                    //CvInvoke.Imshow("Mask", mask.Resize(0.2, Emgu.CV.CvEnum.Inter.Nearest));

                    try
                    {
                        br.DetectAndCompute(theMaster, null, kp_m, des_m, false);
                        br.DetectAndCompute(theTest, null, kp_t, des_t, false);
                        br = null;
                    }
                    catch (CvException ex)
                    {
                        MessageBox.Show(ex.ToString());
                        throw ex;
                    }
                    finally
                    {
                        if (br != null)
                            br.Dispose();
                    }


                    BFMatcher bf = new BFMatcher(DistanceType.L2);
                    VectorOfVectorOfDMatch raw_matches = new Emgu.CV.Util.VectorOfVectorOfDMatch();

                    bf.Add(des_t);

                    try
                    {
                        bf.KnnMatch(des_m, raw_matches, 2, null);
                        bf = null;
                    }
                    catch (Exception e1)
                    {
                        MessageBox.Show(e1.Message);
                        throw;
                    }

                    MDMatch[][] dmatches_tm = raw_matches.ToArrayOfArray();

                    List<MDMatch> good_matches = new List<MDMatch>();
                    foreach (MDMatch[] m in dmatches_tm)
                    {
                        if (m[0].Distance < 0.8 * m[1].Distance && m[0].Distance > 0.7 * m[1].Distance)
                            good_matches.Add(m[0]);
                    }

                    List<int> qidx_t = new List<int>();
                    List<int> tidx_t = new List<int>();
                    int final_dist_t = 0;
                    int oct_count_t = 0;

                    foreach (MDMatch m in good_matches)
                    {
                        var a = kp_m[m.QueryIdx];
                        var b = kp_t[m.TrainIdx];
                        //if(Convert.ToInt32(a.Octave) == Convert.ToInt32(b.Octave))
                        //if (Math.Abs(a.Size - b.Size) < 1f)
                        {
                            oct_count_t += 1;
                            tidx_t.Add(m.TrainIdx);
                            qidx_t.Add(m.QueryIdx);
                        }
                    }

                    for (int i = 0; i < qidx_t.Count; i++)
                    {
                        int q_base = qidx_t[i];
                        int t_base = tidx_t[i];

                        for (int j = i+1; j < qidx_t.Count; j++)
                        {
                            float q = eucledianDist(kp_m[qidx_t[j]].Point, kp_m[q_base].Point);
                            float t = eucledianDist(kp_t[tidx_t[j]].Point, kp_t[t_base].Point);

                            if (Math.Abs(q - t) <= (0.005f * q))
                                final_dist_t++;
                        }
                    }


                    //Image<Bgr, byte> resultImg = new Image<Bgr, byte>(theMaster.Size);
                    //Features2DToolbox.DrawMatches(theMaster, kp_m, theTest, kp_t, raw_matches, resultImg, new MCvScalar(0), new MCvScalar(155));
                    //CvInvoke.Imshow("Res", resultImg.Resize(0.3, Emgu.CV.CvEnum.Inter.Nearest));

                    float raw_len = raw_matches.Size;
                    float final_dist = final_dist_t;
                    float per = (final_dist / raw_len) * 100;
                    float rat = 2000f * (final_dist_t / (float)oct_count_t) / raw_len;
                    float frac = (final_dist / oct_count_t);
                    float f = (final_dist * oct_count_t) / (raw_len); //result
                    long time_taken = timer.ElapsedMilliseconds;
                    var name = test.Substring(35).Split('.')[0];
                    string result = string.Format("{0},{1},{2},{3},{4},{6},{7}", name, raw_len, oct_count_t, final_dist, per, rat, frac, f);
                    return result;
                }
            }

            public float eucledianDist(PointF a, PointF b)
            {
                float x = a.X - b.X;
                float y = a.Y - b.Y;
                return ((float)Math.Sqrt((x * x) + (y * y)));
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            //openFileDialog2.ShowDialog();
            string bg = "";
            //if (openFileDialog2.FileName.ToString() != "")
            //{
                //bg = openFileDialog2.FileName.ToString();
            //}
			
			
            //myClass myobj = new myClass(master, test, bg);
            //myobj.compare();
            //myobj = null;
            GC.Collect();
            GC.WaitForPendingFinalizers();

            
            var allFiles = Directory.GetFiles("D:\\Images\\TVS_EPL_IPS - Copy\\Sproc");
            foreach(string m in allFiles)
            {
                var theResult = new System.Text.StringBuilder();
                master = m;
                foreach (string file in allFiles)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    test = file;
                    if (test != master)
                    {
                        myClass myobj = new myClass(master, test, bg);
                        string result = myobj.compareAll();
                        myobj = null;
                        theResult.AppendLine(result);
                    }
                    
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
                var target = "D:\\result\\" + m.Substring(35).Split('.')[0] + ".csv";
                File.WriteAllText(target, theResult.ToString());

            }

            MessageBox.Show("Completed");

            
        }

    }
}
