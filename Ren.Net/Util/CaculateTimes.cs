using System;
using System.Collections.Generic;
using System.Text;

namespace Ren.Net.Util
{
    public class CaculateTimes
    {
        public int Hour { set; get; }
        public int Min { set; get; }
        public int Second { set; get; }
        public int Ms { set; get; }
        public CaculateTimes()
        {

        }

        public void StartTime()
        {
            Hour = DateTime.Now.Hour;
            Min = DateTime.Now.Minute;
            Second = DateTime.Now.Second;
            Ms = DateTime.Now.Millisecond;
        }
        public string EndTime()
        {
            int hour = DateTime.Now.Hour;
            int min = DateTime.Now.Minute;
            int second = DateTime.Now.Second;
            int ms = DateTime.Now.Millisecond;
            if (hour < Hour)
                hour += 24;
            int millisecond1 = ((hour * 60 + min) * 60 + second) * 1000 + ms;
            int millisecond2 = ((Hour * 60 + Min) * 60 + Second) * 1000 + Ms;
            int time = millisecond1 - millisecond2;
            //+"\t 剩余时间："
            return GetTime(time);
        }
        /// <summary>
        /// 计算剩余时间
        /// </summary>
        /// <param name="index">是剩下还有多少遍</param>
        /// <returns></returns>
        public string EndTime(int index)
        {
            int hour = DateTime.Now.Hour;
            int min = DateTime.Now.Minute;
            int second = DateTime.Now.Second;
            int ms = DateTime.Now.Millisecond;
            if (hour < Hour)
                hour += 24;
            int millisecond1 = ((hour * 60 + min) * 60 + second) * 1000 + ms;
            int millisecond2 = ((Hour * 60 + Min) * 60 + Second) * 1000 + Ms;
            int time = millisecond1 - millisecond2;
            return GetTime(time) + "\t剩余时间：" + GetTime(index * time);
        }
        private string GetTime(int millisecond)
        {
            string line = "";
            line = millisecond % 1000 + "ms";
            millisecond /= 1000;
            line = millisecond % 60 + "s:" + line;
            millisecond /= 60;
            line = millisecond % 60 + "m:" + line;
            millisecond /= 60;
            line = millisecond % 24 + "h:" + line;
            millisecond /= 24;
            line = millisecond + "d:" + line;
            return line;
        }
    }
}
