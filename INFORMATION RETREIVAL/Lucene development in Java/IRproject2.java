//Done by Sriram Venkataramanan
// UB person ID is : 50289666

import java.util.*;
import java.io.*;


import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.MultiFields;
import org.apache.lucene.index.DirectoryReader;

import java.nio.file.*;
import java.nio.file.Paths;

import org.apache.lucene.index.*;

import org.apache.lucene.store.*;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;


public class IRproject2 
{
	//hash map that conatins the terms and its corresonding doc list
	public static HashMap<String,LinkedList<Integer>> hsh_term_postings=new HashMap<String,LinkedList<Integer>>();
	// global variables to print the number of comparisons
	static int compforDatAnd=0;static int compforDatOr=0;
	static int compforTaatAnd=0;static int compforTaatOr=0;

	public static LinkedList<Integer> taatOrfunc(LinkedList<Integer>list1,LinkedList<Integer>list2)
	{	
		int i=0;
		int j=0;

		LinkedList<Integer>listFin=new LinkedList<Integer>();
		
		while(i<(int)list1.size()&&j<(int)list2.size())
		{
			if((int)list1.get(i)==(int)list2.get(j))
			{	
				listFin.add(list1.get(i));
				i++;
				j++;
				compforTaatOr++;
			}
			else if((int)list1.get(i)<(int)list2.get(j))
			{
				listFin.add(list1.get(i));
				i++;
				compforTaatOr++;
			}
			else if((int)list1.get(i)>(int)list2.get(j))
			{	
				 listFin.add(list2.get(j));
				j++;
				compforTaatOr++;
			}
		}
		
		while(i<(int)list1.size())
		{
			listFin.add(list1.get(i));
			i++;
		}
		while(j<(int)list2.size())
		{
			listFin.add(list2.get(j));
			j++;
		}
		return listFin;		
				
	}
	
	public static LinkedList<Integer> taatAndfunc(LinkedList<Integer>list1,LinkedList<Integer>list2)
	{	
		int i=0;
		int j=0;
		LinkedList<Integer>listfinal=new LinkedList<Integer>();
		while(i<(int)list1.size()&&j<(int)list2.size())
		{
			if((int)list1.get(i)==(int)list2.get(j))
			{	
				listfinal.add(list1.get(i));
				i++;
				j++;
				compforTaatAnd++;
			}
			else if((int)list1.get(i)<(int)list2.get(j))
			{
				i++;
				compforTaatAnd++;
			}
			else if((int)list1.get(i)>(int)list2.get(j))
			{					
				j++;
				compforTaatAnd++;
			}
		}
		return listfinal;
	}
	
	public static LinkedList<Integer> daatAndfunc(LinkedList<Integer>[] listfull,int size)
	{
		int i=0;int comp=0;int flagforCheck=0; int countTotal=0;
		LinkedList<Integer>listfinal=new LinkedList<Integer>();
		
		int[] ptrforlist=new int[size];
		
		for(i=0;i<size;i++)
			ptrforlist[i]=0;
		
		int gmsize=listfull[0].get(0);
		
		while(flagforCheck==0)
		{
			countTotal=0;
			for(i=0;i<size;i++)
			{				
				if(((int)(listfull[i].get(ptrforlist[i])))==comp)
				{
					if(i!=0)
						compforDatAnd++;
					countTotal++;
				}
				else if(((int)(listfull[i].get(ptrforlist[i])))>comp)
				{
					comp=listfull[i].get(ptrforlist[i]);
					compforDatAnd++;
				}
				else 
				{
					ptrforlist[i]++;
					compforDatAnd++;
				}
			
			if(countTotal==size)
			{
				listfinal.add(comp);
				for(int k=0;k<size;k++)
				{
					ptrforlist[k]++;
					if(ptrforlist[k]>=(listfull[k].size()))
					{
						flagforCheck=1;
						break;
					}
				}
				
			}
			if(ptrforlist[i]==(int)listfull[i].size())
			{
				flagforCheck=1;
				break;
			}

		}
		}	
		
		return listfinal;
	}
		
	
	public static LinkedList<Integer> daatOrfunc(LinkedList<Integer>[] listfull,int size)
	{
		int i=0;int flagtoCheck=0;int min_value=100000000;int countTotal=0;int ptrlist=0;
		LinkedList<Integer>listfinal=new LinkedList<Integer>();
		int[] ptrforlist=new int[size];

		for(i=0;i<size;i++)
		{
			ptrforlist[i]=0;
		}
		int gmsize=listfull[0].get(0);
		
		while(flagtoCheck==0)
		{
			countTotal=0;
			for(i=0;i<size;i++)
			{				
				
				if(ptrforlist[i]!=(int)listfull[i].size())
				{
					if(((int)(listfull[i].get(ptrforlist[i])))<min_value)
					{
						min_value=listfull[i].get(ptrforlist[i]);
						ptrlist=i;
						compforDatOr++;
					}
					else if(((int)(listfull[i].get(ptrforlist[i])))==min_value)
					{
						if(i==0)
							compforDatOr++;
						ptrforlist[i]++;
						countTotal++;
						
					}
				}
				else
					countTotal++;
				}
			if(countTotal==size)
			{
				flagtoCheck=1;
				break;
			}
			listfinal.add(min_value);
			ptrforlist[ptrlist]++; min_value=99999999; 
			
		}
		return listfinal;
	}
	
	//method to obtain all the postings of a term
	public static LinkedList<Integer> allPostings(String st) 
	{ 
		LinkedList<Integer>ltemp=new LinkedList<Integer>();
		//retreiving the posting for a particular query term from hashmap
		ltemp=hsh_term_postings.get(st);
		
		return ltemp;
	}
			
	public static void main(String[] args)throws IOException,NullPointerException
	{   
		//getting index input and output files as command line arguments
		String pathOfIndex = args[0];
		String pathOfoutput = args[1];
		String pathOfinput = args[2];
		
		//getting the path of index
		//Path docpath=Paths.get("E:\\Sriram\\index");
		Path docpath=Paths.get(pathOfIndex);
		IndexReader reader = DirectoryReader.open(FSDirectory.open(docpath));
		
		// Scanning to read the terms from the input document
		//Scanner scfile = new Scanner(new File("E:\\Sriram\\input.txt"));
	    Scanner scfile = new Scanner(new File(pathOfinput),"UTF-8");
        
	    //output file where the results are printed 
		PrintWriter writer = new PrintWriter(pathOfoutput, "UTF-8");
		
		int kcount=0;
		
		Collection<String>fullfile=MultiFields.getIndexedFields(reader);
		
		for(String strofterm:fullfile)
		{	
			if(!(strofterm.equalsIgnoreCase("id")||strofterm.equalsIgnoreCase("_version_")))
			{
		
				Terms allTerms = MultiFields.getTerms(reader,strofterm);
				TermsEnum itrate=allTerms.iterator();
		
				while(itrate.next()!=null)
				{
					LinkedList<Integer>listposting=new LinkedList<>();
					ArrayList<Integer>p_list=new ArrayList<>();
				
					PostingsEnum postdocument=itrate.postings(null);
					while(postdocument.nextDoc()!=PostingsEnum.NO_MORE_DOCS)
					{
						listposting.add(postdocument.docID());
						kcount++;
					}
					//storing all the terms and documents ids in a hashmap
					hsh_term_postings.put(itrate.term().utf8ToString(),listposting);
				}
			}
		}
		System.out.println("Code running Successfully");
		System.out.println("Total number of terms in the index: "+hsh_term_postings.size());
		
		System.out.println("The result is printed in the output file");
		
			// Storing the query terms that are entered in the input file in an array list
			while(scfile.hasNextLine())
			{
				ArrayList<String>listofQueryterms=new ArrayList<String>();

				 Scanner s2 = new Scanner(scfile.nextLine());
			        while (s2.hasNext()) 
			        {
			            String queryterm = s2.next();
			           // the array list has all the query terms
			            listofQueryterms.add(queryterm);
			        }
			
			LinkedList<Integer>[] listTC = new LinkedList[listofQueryterms.size()];
				
					for(int i=0;i<listofQueryterms.size();i++)
			{
				writer.println("GetPostings");	
				
				writer.println(listofQueryterms.get(i));
				
				listTC[i]=allPostings(listofQueryterms.get(i));
			
			writer.print("Postings list: ");
		
				for(int j=0;j<listTC[i].size();j++)
				{
					writer.print(listTC[i].get(j)+ " ");
					
				}
				writer.println();
			}
			
			LinkedList<Integer>tem=new LinkedList();
			LinkedList<Integer>tem1=new LinkedList();
			LinkedList<Integer>tem2=new LinkedList();
			LinkedList<Integer>tem3=new LinkedList();
            tem=(LinkedList)listTC[0].clone();
			tem1=(LinkedList)listTC[0].clone();
			tem2=(LinkedList)listTC[0].clone();
			tem3=(LinkedList)listTC[0].clone();


			//performing the operations of TaatAnd
			writer.println("TaatAnd");
			for(int i=0;i<listofQueryterms.size();i++)	
			{
			writer.print(listofQueryterms.get(i)+ " ");
			}
			writer.println();
			for(int i=1;i<listofQueryterms.size();i++)
			{
				tem=(LinkedList)taatAndfunc(tem,listTC[i]);
			}
			
			writer.print("Results: ");
			if(tem.size()==0)
			{
				writer.print("empty");
			}
			else
			{
			for(int i=0;i<tem.size();i++)
			{
				writer.print(tem.get(i)+ " ");
			}
			}
			
			
			writer.println();
			writer.println("Number of documents in results: "+tem.size());
			writer.println("Number of comparisons: " +compforTaatAnd);
			
			//performing the operations of taator
			writer.println("TaatOr");
			
			for(int i=0;i<listofQueryterms.size();i++)	
			{
			writer.print(listofQueryterms.get(i)+ " ");
			}
			writer.println();

			for(int i=1;i<listofQueryterms.size();i++)
			{
				tem1=(LinkedList)taatOrfunc(tem1,listTC[i]);
			}
			
			writer.print("Results: ");
			if(tem1.size()==0)
			{
				writer.print("empty");
			}
			else
			{
			for(int i=0;i<tem1.size();i++)
			{
				writer.print(tem1.get(i)+ " ");
			}
			}
			writer.println();
			
			
		    writer.println("Number of documents in results: "+tem1.size());
		    writer.println("Number of comparisons: " +compforTaatOr);
			
		    //performingn the operations of Daatand
			writer.println("DaatAnd");
			
			for(int i=0;i<listofQueryterms.size();i++)	
			{
			writer.print(listofQueryterms.get(i)+ " ");
			}
			writer.println();

			tem2=daatAndfunc(listTC,listofQueryterms.size());

			
			writer.print("Results:");
			if(tem2.size()==0)
			{
				writer.print("empty");
			}
			else
			{
			for(int i=0;i<tem2.size();i++)
			{
				writer.print(tem2.get(i)+ " ");
			}
			}
            writer.println();
			
			
			writer.println("Number of documents in results: "+tem2.size());
			writer.println("Number of comparisons: " +compforDatAnd);
			
			//performingn the operations of Daator
			writer.println("DaatOr");
			for(int i=0;i<listofQueryterms.size();i++)	
			{
			writer.print(listofQueryterms.get(i)+ " ");
			}
			writer.println();

			tem3=daatOrfunc(listTC,listofQueryterms.size());

			
			writer.print("Results: ");
			if(tem3.size()==0)
			{
				writer.print("empty");
			}
			else
			{
			for(int i=0;i<tem3.size();i++)
			{
				writer.print(tem3.get(i)+ " ");
			}
			}
            writer.println();
			

			writer.println("Number of documents in results: "+tem3.size());
			writer.println("Number of comparisons: " +compforDatOr);
            
			reader.close();
			
		}
			compforDatAnd=0;
			compforTaatOr=0;
			compforTaatAnd=0;
			compforDatOr=0;
			writer.close();
	}
}

		
		
	
