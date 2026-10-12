import type { Metadata, Viewport } from "next";
import { Bricolage_Grotesque, Figtree, JetBrains_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

const bricolage = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin"],
});

const figtree = Figtree({
  variable: "--font-figtree",
  subsets: ["latin"],
});

// Used only inside code and raw tool output.
const code = JetBrains_Mono({
  variable: "--font-code",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "RSI Science Assistant",
  description: "Ask a research question and watch the agent's trajectory.",
};

export const viewport: Viewport = {
  themeColor: "#0B0D1E",
  colorScheme: "dark",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`dark ${bricolage.variable} ${figtree.variable} ${code.variable} h-full antialiased`}
    >
      <body className="h-full overflow-hidden">
        <TooltipProvider delay={250}>{children}</TooltipProvider>
      </body>
    </html>
  );
}
