using Ren.Device;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Ren.Net.Util
{
    public class InstenceHelper<T> where T : class
    {
        public static List<T> GetInstence(Type typeClass, object[] @param)
        {
            var types = typeClass.Assembly.GetTypes().Where(type => type.IsSubclassOf(typeClass));
            List<T> linears = new List<T>();
            foreach (var item in types)
            {
                linears.Add(Activator.CreateInstance(item, @param) as T);
            }
            return linears;
        }
    }
}
